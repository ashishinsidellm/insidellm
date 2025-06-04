#!/usr/bin/env python3
"""
InsideLLM SDK Example Runner

This script helps you run and test the SDK examples with your actual API credentials.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv("INSIDELLM_API_KEY")
    if not api_key:
        print("Missing API Key Configuration")
        print("=" * 50)
        print("Before running examples, you need to set your InsideLLM API key:")
        print()
        print("Option 1: Environment variable")
        print("export INSIDELLM_API_KEY='your-api-key-here'")
        print()
        print("Option 2: Create .env file")
        print("echo 'INSIDELLM_API_KEY=your-api-key-here' > .env")
        print()
        return False
    
    print(f"API Key found: {api_key[:8]}...{api_key[-4:]}")
    return True

def test_sdk_installation():
    """Test if SDK is properly installed"""
    try:
        import insidellm
        print(f"SDK installed successfully (version: {insidellm.__version__})")
        return True
    except ImportError as e:
        print(f"SDK installation issue: {e}")
        print("Please install the SDK first:")
        print("pip install -e .")
        return False

def run_example(example_name, description):
    """Run a specific example"""
    print(f"\nRunning: {example_name}")
    print(f"Description: {description}")
    print("-" * 60)
    
    example_path = f"examples/{example_name}"
    if not os.path.exists(example_path):
        print(f"Example file not found: {example_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, example_path], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
        else:
            print("FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT - Example took too long to run")
        return False
    except Exception as e:
        print(f"ERROR - {e}")
        return False

def quick_api_test():
    """Quick test to verify API connectivity"""
    print("\nQuick API Connectivity Test")
    print("-" * 40)
    
    test_code = '''
import sys
sys.path.insert(0, ".")
import insidellm
import os

try:
    # Initialize SDK
    insidellm.initialize(api_key=os.getenv("INSIDELLM_API_KEY"))
    
    # Test basic functionality
    client = insidellm.get_client()
    run_id = client.start_run(user_id="api-test-user")
    
    # Log a test event
    event = insidellm.Event.create_user_input(
        run_id=run_id,
        user_id="api-test-user",
        input_text="API connectivity test"
    )
    client.log_event(event)
    
    # Flush immediately to test API
    client.flush()
    
    # Check health
    healthy = client.is_healthy()
    
    client.end_run(run_id)
    insidellm.shutdown()
    
    print(f"API Test Result: {'SUCCESS' if healthy else 'FAILED'}")
    
except Exception as e:
    print(f"API Test Result: FAILED - {e}")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout if result.stdout else result.stderr)
        return "SUCCESS" in (result.stdout or "")
    except Exception as e:
        print(f"API test failed: {e}")
        return False

def main():
    """Main runner function"""
    print("InsideLLM SDK Example Runner")
    print("=" * 50)
    
    # Check prerequisites
    if not check_api_key():
        return
    
    if not test_sdk_installation():
        return
    
    # Quick API test
    api_works = quick_api_test()
    if not api_works:
        print("\nAPI connectivity test failed. Please check:")
        print("1. Your API key is correct")
        print("2. You have internet connectivity")
        print("3. The InsideLLM service is accessible")
        print("\nContinuing with examples anyway...")
    
    # Available examples
    examples = [
        ("basic_usage.py", "Core SDK functionality with manual event creation"),
        ("custom_agent_example.py", "Custom agent integration with decorators and context managers"),
        ("langchain_example.py", "LangChain integration (requires langchain package)")
    ]
    
    print(f"\nAvailable Examples ({len(examples)} total)")
    print("=" * 50)
    
    # Run examples
    results = {}
    for example, description in examples:
        success = run_example(example, description)
        results[example] = success
        time.sleep(1)  # Brief pause between examples
    
    # Summary
    print(f"\nExample Run Summary")
    print("=" * 50)
    
    for example, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status} {example}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} examples passed")
    
    if success_count == total_count:
        print("\nAll examples completed successfully!")
        print("Your SDK is ready for production use.")
    else:
        print(f"\n{total_count - success_count} examples failed.")
        print("Check the error messages above for troubleshooting.")
    
    print("\nNext Steps:")
    print("1. Review the examples that worked")
    print("2. Integrate the SDK patterns into your own code")
    print("3. Monitor your analytics in the InsideLLM dashboard")
    print("4. Scale up to production workloads")

if __name__ == "__main__":
    main()