import os
import time
import random
import insidellm
from typing import Dict, Any, List, Optional
import json

extract_prompt_fn = lambda self_instance, prompt, **kwargs: prompt # self_instance is the agent
extract_response_fn = lambda response: response

class FlightBookingAgent:
    def __init__(self, user_session_id: str):
        self.user_session_id = user_session_id
        self.current_search_results: List[Dict[str, Any]] = []
        self.selected_flight: Optional[Dict[str, Any]] = None
        self.booking_details: Dict[str, Any] = {}
        self.agent_name = "FlightBookingAgent"
        self.model_name = "flight-booking-llm-v1"
        self.llm_provider = "mock_provider"

    @insidellm.track_llm_call(
        model_name="flight-booking-llm-v1",
        provider="mock_provider",
        extract_prompt=extract_prompt_fn,
        extract_response=extract_response_fn
    )
    def _interact_with_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        time.sleep(random.uniform(0.01, 0.02)) # Minimal sleep for tests
        if "search flights" in prompt.lower():
            origin, destination, date = "London", "New York", "2024-12-24"
            try:
                if "from" in prompt and "to" in prompt:
                    parts = prompt.lower().split("from ")[1]
                    origin = parts.split(" to ")[0].strip()
                    if " on " in parts.split(" to ")[1]:
                        destination_part = parts.split(" to ")[1]
                        destination = destination_part.split(" on ")[0].strip()
                        date = destination_part.split(" on ")[1].split()[0].strip()
                    else:
                        destination = parts.split(" to ")[1].strip()
            except Exception: pass
            response = {"intent": "search_flights", "parameters": {"origin": origin, "destination": destination, "date": date}}
        elif "select flight" in prompt.lower():
            flight_id = "FL123"
            try: flight_id = prompt.split()[-1].strip()
            except Exception: pass
            response = {"intent": "select_flight", "parameters": {"flight_id": flight_id}}
        elif "my name is" in prompt.lower() or "passenger details" in prompt.lower() or "book flight" in prompt.lower() :
            name, email = "John Doe", "johndoe@example.com"
            if "my name is" in prompt.lower():
                try:
                    name_part = prompt.lower().split("my name is ")[1]
                    name = name_part.split(",")[0].strip()
                    if "email" in prompt.lower():
                         email = prompt.lower().split("email ")[1].strip()
                except Exception: pass
            response = {"intent": "gather_passenger_details", "parameters": {"name": name, "email": email}}
        elif "payment" in prompt.lower():
            response = {"intent": "process_payment", "parameters": {"payment_method": "credit_card", "card_number_last4": "1234"}}
        elif "confirm" in prompt.lower():
            response = {"intent": "confirm_booking", "parameters": {}}
        else:
            response = {"intent": "unknown", "parameters": {}, "original_query": prompt}
        return response

    @insidellm.track_tool_use(
        tool_name="external_api_dispatcher",
        tool_type="api",
        # Corrected lambda signature to match how decorator calls it with *args, **kwargs from decorated method
        extract_parameters=lambda self_instance, tool_name, api_endpoint, parameters, **other_kwargs: {
            "specific_tool_name": tool_name, # tool_name is an arg to _call_external_api
            "api_endpoint": api_endpoint,   # api_endpoint is an arg to _call_external_api
            "original_parameters": parameters # parameters is an arg to _call_external_api
        },
        extract_response=extract_response_fn
    )
    def _call_external_api(self, tool_name: str, api_endpoint: str, parameters: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        time.sleep(random.uniform(0.01, 0.02)) # Minimal sleep for tests
        mock_response = {}
        if tool_name == "flight_search_api":
            mock_response = {
                "data": [
                    {"flight_id": "FL123", "airline": "MockAir", "price": 300 + random.randint(-5, 5), "departure": parameters.get("origin", "N/A"), "arrival": parameters.get("destination", "N/A"), "date": parameters.get("date", "N/A")},
                    {"flight_id": "FL456", "airline": "FlyMock", "price": 350 + random.randint(-5, 5), "departure": parameters.get("origin", "N/A"), "arrival": parameters.get("destination", "N/A"), "date": parameters.get("date", "N/A")},
                ], "count": 2}
        elif tool_name == "payment_processing_api":
            price = parameters.get("amount", random.randint(250,500))
            mock_response = {"status": "success" if random.choice([True,True,False]) else "failed", "transaction_id": f"txn_{random.randint(10000,99999)}", "amount_processed": price}
        elif tool_name == "booking_confirmation_api":
            mock_response = {"status": "confirmed", "booking_id": f"bkng_{random.randint(10000,99999)}", "details": parameters}
        else:
            mock_response = {"error": "Unknown API or tool"}
        return mock_response

    @insidellm.track_agent_step("flight_search_initiated")
    def search_flights(self, search_query_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        api_response = self._call_external_api(tool_name="flight_search_api", api_endpoint="flights/search", parameters=search_query_details)
        self.current_search_results = api_response.get("data", [])
        return self.current_search_results

    @insidellm.track_agent_step("flight_selection")
    def select_flight(self, flight_id: str) -> Optional[Dict[str, Any]]:
        for flight in self.current_search_results:
            if flight["flight_id"] == flight_id:
                self.selected_flight = flight
                return self.selected_flight
        return None

    @insidellm.track_agent_step("passenger_details_collection")
    def gather_passenger_details(self, details: Dict[str, Any]) -> bool:
        if not self.selected_flight: return False
        self.booking_details["passenger_info"] = details
        return True

    @insidellm.track_agent_step("payment_processing")
    def process_payment(self, payment_info: Dict[str, Any]) -> bool:
        if not self.selected_flight or "passenger_info" not in self.booking_details: return False
        payment_params = {"amount": self.selected_flight.get("price"), "currency": "USD", "payment_details": payment_info}
        api_response = self._call_external_api(tool_name="payment_processing_api", api_endpoint="payments/process", parameters=payment_params)
        if api_response.get("status") == "success":
            self.booking_details["payment_status"] = "Paid"; self.booking_details["transaction_id"] = api_response.get("transaction_id")
            return True
        self.booking_details["payment_status"] = "Failed"; return False

    @insidellm.track_agent_step("booking_confirmation")
    def confirm_booking(self) -> Optional[Dict[str, Any]]:
        if not (self.selected_flight and self.booking_details.get("passenger_info") and self.booking_details.get("payment_status") == "Paid"): return None
        confirmation_params = {"flight_id": self.selected_flight["flight_id"], "passenger_details": self.booking_details["passenger_info"], "transaction_id": self.booking_details.get("transaction_id")}
        api_response = self._call_external_api(tool_name="booking_confirmation_api", api_endpoint="bookings/confirm", parameters=confirmation_params)
        if api_response.get("status") == "confirmed":
            self.booking_details["confirmation_id"] = api_response.get("booking_id"); self.booking_details["status"] = "Confirmed"
            return {"confirmation_id": self.booking_details['confirmation_id'], "flight_details": self.selected_flight, "passenger_info": self.booking_details["passenger_info"], "total_paid": self.selected_flight.get("price")}
        self.booking_details["status"] = "Confirmation Failed"; return None

    def _process_query_with_tracker(self, query: str, user_id: str, tracker: insidellm.InsideLLMTracker) -> str:
        try:
            tracker.log_user_input(input_text=query)
            llm_response = self._interact_with_llm(prompt=f"Parse user query for flight booking: {query}")
            intent = llm_response.get("intent")
            parameters = llm_response.get("parameters", {})
            agent_response_str = f"Could not process intent: {intent}"
            if intent == "search_flights":
                results = self.search_flights(search_query_details=parameters); agent_response_str = f"Found {len(results)} flights: {json.dumps(results, indent=2)}"
            elif intent == "select_flight":
                flight_id = parameters.get("flight_id")
                if flight_id: flight = self.select_flight(flight_id); agent_response_str = f"Selected flight: {json.dumps(flight, indent=2)}" if flight else f"Could not select flight {flight_id}."
                else: agent_response_str = "Please specify a flight ID to select."
            elif intent == "gather_passenger_details":
                if self.selected_flight:
                    details_to_gather = parameters if parameters.get("name") != "John Doe" else {"name": "Jane Flyer", "contact": "jane.flyer@example.com", "dob": "1990-01-01"}
                    if self.gather_passenger_details(details_to_gather): agent_response_str = f"Passenger details gathered for {details_to_gather.get('name')}. Please provide payment information."
                    else: agent_response_str = "Could not gather passenger details. Ensure a flight is selected."
                else: agent_response_str = "Please select a flight before providing passenger details."
            elif intent == "process_payment":
                if self.selected_flight and self.booking_details.get("passenger_info"):
                    mock_payment_info = parameters if parameters.get("payment_method") else {"card_type": "Visa", "last4": "1111"}
                    if self.process_payment(mock_payment_info): agent_response_str = "Payment successful. Ready to confirm booking."
                    else: agent_response_str = "Payment failed. Please try again or use a different payment method."
                else: agent_response_str = "Please select a flight and provide passenger details before payment."
            elif intent == "confirm_booking":
                confirmation = self.confirm_booking(); agent_response_str = f"Booking confirmed! Details: {json.dumps(confirmation, indent=2)}" if confirmation else "Booking could not be confirmed."
            else: agent_response_str = f"Sorry, I didn't understand that. My LLM reported intent: {intent} with params: {parameters}."
            tracker.log_agent_response(response_text=agent_response_str)
            return agent_response_str
        except Exception as e:
            tracker.log_error(error_type=type(e).__name__, error_message=str(e)); import traceback; traceback.print_exc() # Print stack trace for live debugging
            return f"An error occurred: {e}"

    def handle_user_query(self, query: str, user_id: str, external_tracker: Optional[insidellm.InsideLLMTracker] = None) -> str:
        if external_tracker: return self._process_query_with_tracker(query, user_id, external_tracker)
        else:
            with insidellm.InsideLLMTracker(user_id=user_id, metadata={"agent_name": self.agent_name, "session_id": self.user_session_id, "turn_description": "FlightBookingTurn"}) as new_tracker:
                return self._process_query_with_tracker(query, user_id, new_tracker)

def main():
    insidellm.initialize(api_key=os.getenv("INSIDELLM_API_KEY", "demo-key-for-testing"), local_testing=True, auto_flush_interval=0.5, batch_size=1) # Faster flush for tests
    agent = FlightBookingAgent(user_session_id="user_session_journey_example_main_func")
    test_user_id = "test_user_journey_main_func_001"
    with insidellm.InsideLLMTracker(user_id=test_user_id, metadata={"agent_name": agent.agent_name, "session_id": agent.user_session_id, "journey_type": "simulated_test_main_func", "run_description": "CompleteFlightBookingJourneyFromMain"}) as main_tracker:
        queries = [
            "search flights from Paris to Tokyo on 2024-11-15",
            f"select flight {agent.current_search_results[0]['flight_id']}" if agent.current_search_results else "select flight FLXYZ", # Ensure selection happens
            "My name is Jean Luc, email jean.luc@starfleet.com, I want to book this flight.",
            "process payment with my Federation Express card ending 7890",
            "confirm my booking please"
        ]

        # Simplified loop to ensure all queries run, conditions handled by agent logic
        for i, q in enumerate(queries):
            if i == 1 and not agent.current_search_results: # Specific logic for select flight if search failed
                 print("Skipping select flight as no results from search_flights")
                 agent.handle_user_query("select flight FLBAD", user_id=test_user_id, external_tracker=main_tracker) # Log a dummy selection
                 continue
            if i == 1 and agent.current_search_results : # update query for select flight
                q = f"select flight {agent.current_search_results[0]['flight_id']}"

            print(f"Executing query {i+1}: {q}")
            agent.handle_user_query(q, user_id=test_user_id, external_tracker=main_tracker)
            # Conditional logic for subsequent steps can be inside agent or here if needed for test flow
            if i == 3 and agent.booking_details.get("payment_status") != "Paid":
                print("Payment failed, not attempting confirmation query.")
                # break # Optional: stop if payment fails, or let it try to confirm and fail

    insidellm.flush()
    insidellm.shutdown()

if __name__ == "__main__":
    main()
