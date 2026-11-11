
import os
import hashlib
import getpass

salt = os.urandom(16).hex()

password = getpass.getpass("Masukkan password admin: ")

hashed = hashlib.pbkdf2_hmac(
    "sha256",
    password.encode("utf-8"),
    bytes.fromhex(salt),
    310000
).hex()

result = f"pbkdf2_sha256$310000${salt}${hashed}"

print("\nPassword hash berhasil dibuat:")
print(result)
