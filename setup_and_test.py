#!/usr/bin/env python3
"""
InsideLLM SDK Setup and Test Script

Quick setup and testing for the InsideLLM Python SDK
"""

import os
import sys
import subprocess
import getpass

def setup_api_key():
    """Interactive API key setup"""
    print("InsideLLM SDK Setup")
    print("=" * 30)
    
    current_key = os.getenv("INSIDELLM_API_KEY")
    if current_key:
        print(f"Current API key: {current_key[:8]}...{current_key[-4:]}")
        use_current = input("Use current API key? (y/n): ").lower().strip()
        if use_current == 'y':
            return current_key
    
    print("\nTo get your API key:")
    print("1. Log into your InsideLLM dashboard")
    print("2. Go to Settings > API Keys")
    print("3. Create or copy your API key")
    
    api_key = getpass.getpass("\nEnter your InsideLLM API key: ").strip()
    
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)
    
    # Set environment variable for this session
    os.environ["INSIDELLM_API_KEY"] = api_key
    
    # Offer to save to .env file
    save_env = input("Save API key to .env file? (y/n): ").lower().strip()
    if save_env == 'y':
        with open('.env', 'w') as f:
            f.write(f"INSIDELLM_API_KEY={api_key}\n")
        print("API key saved to .env file")
    
    return api_key

def test_api_connection(api_key):
    """Test API connection"""
    print("\nTesting API connection...")
    
    test_script = f'''
import sys
import os
sys.path.insert(0, ".")

# Set API key
os.environ["INSIDELLM_API_KEY"] = "{api_key}"

try:
    import insidellm
    
    # Initialize
    insidellm.initialize(api_key="{api_key}")
    client = insidellm.get_client()
    
    # Quick test
    run_id = client.start_run(user_id="setup-test")
    
    event = insidellm.Event.create_user_input(
        run_id=run_id,
        user_id="setup-test", 
        input_text="Setup test message"
    )
    client.log_event(event)
    client.flush()
    
    client.end_run(run_id)
    insidellm.shutdown()
    
    print("SUCCESS: API connection working")
    
except Exception as e:
    print(f"FAILED: {{e}}")
'''
    
    result = subprocess.run([sys.executable, "-c", test_script], 
                          capture_output=True, text=True)
    
    if "SUCCESS" in result.stdout:
        print("✓ API connection successful")
        return True
    else:
        print("✗ API connection failed")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_basic_example():
    """Run the basic usage example"""
    print("\nRunning basic example...")
    
    try:
        result = subprocess.run([sys.executable, "examples/basic_usage.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Basic example completed successfully")
            return True
        else:
            print("✗ Basic example failed")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Basic example timed out")
        return False
    except Exception as e:
        print(f"✗ Error running basic example: {e}")
        return False

def main():
    """Main setup and test function"""
    
    # Setup API key
    api_key = setup_api_key()
    
    # Test connection
    if not test_api_connection(api_key):
        print("\nPlease check:")
        print("1. Your API key is correct")
        print("2. You have internet access")
        print("3. The InsideLLM service is available")
        return
    
    # Run basic example
    run_basic_example()
    
    print("\nSetup Complete!")
    print("=" * 30)
    print("Your SDK is ready to use. Next steps:")
    print("1. Run: python run_examples.py")
    print("2. Check examples/ directory for integration patterns")
    print("3. Review TESTING_GUIDE.md for detailed instructions")
    print("4. Start integrating with your agent code")

if __name__ == "__main__":
    main()