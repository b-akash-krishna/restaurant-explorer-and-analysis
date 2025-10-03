"""
Debug script to test password hashing functionality
Run this to identify the exact issue with bcrypt
"""
from passlib.context import CryptContext

# Test different configurations
print("Testing bcrypt configurations...\n")

# Configuration 1: Default (this is likely causing the error)
try:
    pwd_context_default = CryptContext(schemes=["bcrypt"], deprecated="auto")
    test_pass = "testpassword"
    hashed = pwd_context_default.hash(test_pass)
    verified = pwd_context_default.verify(test_pass, hashed)
    print(f"✓ Default config works: {verified}")
except Exception as e:
    print(f"✗ Default config failed: {e}")

# Configuration 2: With truncate_error=False
try:
    pwd_context_truncate = CryptContext(
        schemes=["bcrypt"], 
        deprecated="auto",
        truncate_error=False
    )
    test_pass = "testpassword"
    hashed = pwd_context_truncate.hash(test_pass)
    verified = pwd_context_truncate.verify(test_pass, hashed)
    print(f"✓ Truncate config works: {verified}")
except Exception as e:
    print(f"✗ Truncate config failed: {e}")

# Configuration 3: Test with long password
try:
    long_password = "a" * 100  # 100 character password
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed = pwd_context.hash(long_password)
    print(f"✗ Long password should have failed but didn't")
except Exception as e:
    print(f"✓ Long password correctly raises error: {type(e).__name__}")

# Configuration 4: Test with manual truncation
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    long_password = "a" * 100
    # Truncate to 72 bytes
    truncated = long_password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    hashed = pwd_context.hash(truncated)
    verified = pwd_context.verify(truncated, hashed)
    print(f"✓ Manual truncation works: {verified}")
except Exception as e:
    print(f"✗ Manual truncation failed: {e}")

# Check passlib version
try:
    import passlib
    print(f"\nPasslib version: {passlib.__version__}")
except:
    print("\nCouldn't determine passlib version")

# Test the actual password from tests
print("\n" + "="*50)
print("Testing actual test password:")
try:
    pwd_context = CryptContext(
        schemes=["bcrypt"], 
        deprecated="auto",
        truncate_error=False
    )
    test_password = "testpassword"
    print(f"Password: '{test_password}'")
    print(f"Length: {len(test_password)} chars, {len(test_password.encode('utf-8'))} bytes")
    hashed = pwd_context.hash(test_password)
    print(f"✓ Hash successful: {hashed[:20]}...")
    verified = pwd_context.verify(test_password, hashed)
    print(f"✓ Verify successful: {verified}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()