"""
Comprehensive Security Manager
"""

import logging
import hashlib
import hmac
import secrets
import json
import re
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from src.tools.config_loader import load_config

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT not available. JWT authentication will be limited.")

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography not available. Encryption will be limited.")

logger = logging.getLogger(__name__)


class SecurityManager:
    """Comprehensive security manager for Pinnacle AI"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)

        # Initialize security components
        self._initialize_encryption()
        self._initialize_authentication()
        self._initialize_input_validation()
        self._initialize_audit_logging()

    def _initialize_encryption(self):
        """Initialize encryption components"""
        encryption_config = self.config.get("security", {}).get("encryption", {})
        key = encryption_config.get("key")

        if not key and CRYPTOGRAPHY_AVAILABLE:
            # Generate new key
            key = Fernet.generate_key().decode()
            self.logger.warning("Generated new encryption key. Store this securely!")
        elif key and CRYPTOGRAPHY_AVAILABLE:
            # Ensure key is bytes
            if isinstance(key, str):
                key = key.encode()
        else:
            key = None

        if key and CRYPTOGRAPHY_AVAILABLE:
            self.cipher = Fernet(key)
            self.data_encryption = DataEncryption(self.cipher)
            self.secure_comm = SecureCommunication(self.cipher)
        else:
            self.cipher = None
            self.data_encryption = None
            self.secure_comm = None

    def _initialize_authentication(self):
        """Initialize authentication components"""
        auth_config = self.config.get("security", {}).get("authentication", {})

        # JWT configuration
        self.jwt_secret = auth_config.get("jwt_secret")
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_hex(32)
            self.logger.warning("Generated new JWT secret. Store this securely!")

        self.jwt_algorithm = auth_config.get("jwt_algorithm", "HS256")
        self.jwt_expiration = auth_config.get("jwt_expiration", 3600)  # 1 hour

        # Initialize authentication
        if JWT_AVAILABLE:
            self.auth = Authentication(
                self.jwt_secret,
                self.jwt_algorithm,
                self.jwt_expiration
            )
        else:
            self.auth = None

        # Initialize API keys
        self.api_keys = APIKeyManager(auth_config.get("api_keys", {}))

    def _initialize_input_validation(self):
        """Initialize input validation components"""
        self.input_validator = InputValidator()

    def _initialize_audit_logging(self):
        """Initialize audit logging"""
        audit_config = self.config.get("security", {}).get("audit", {})
        self.audit_logger = AuditLogger(
            log_file=audit_config.get("log_file", "audit.log"),
            retention_days=audit_config.get("retention_days", 30)
        )

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.data_encryption:
            return self.data_encryption.encrypt(data)
        return data  # No encryption available

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.data_encryption:
            return self.data_encryption.decrypt(encrypted_data)
        return encrypted_data  # No decryption available

    def generate_token(self, user_id: str, roles: List[str]) -> str:
        """Generate authentication token"""
        if self.auth:
            return self.auth.generate_token(user_id, roles)
        return "token_not_available"

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate authentication token"""
        if self.auth:
            return self.auth.validate_token(token)
        return None

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return self.api_keys.validate(api_key)

    def validate_input(self, input_data: str, input_type: str) -> bool:
        """Validate user input"""
        return self.input_validator.validate(input_data, input_type)

    def log_audit_event(self, event_type: str, user_id: str, details: Dict):
        """Log audit event"""
        self.audit_logger.log(event_type, user_id, details)

    def secure_request(self, request: Dict) -> Dict:
        """Secure an API request"""
        if self.secure_comm:
            return self.secure_comm.secure_request(request)
        return request  # No encryption available

    def verify_request(self, request: Dict) -> bool:
        """Verify a secure request"""
        if self.secure_comm:
            return self.secure_comm.verify_request(request)
        return True  # No verification available


class DataEncryption:
    """Data encryption and decryption"""

    def __init__(self, cipher: Fernet):
        self.cipher = cipher
        self.logger = logging.getLogger(__name__)

    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        try:
            if not isinstance(data, str):
                data = str(data)
            encrypted = self.cipher.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise


class SecureCommunication:
    """Secure communication between components"""

    def __init__(self, cipher: Fernet):
        self.cipher = cipher
        self.logger = logging.getLogger(__name__)

    def secure_request(self, request: Dict) -> Dict:
        """Secure a request with encryption and signature"""
        try:
            # Serialize request
            request_str = json.dumps(request)

            # Encrypt
            encrypted = self.cipher.encrypt(request_str.encode())

            # Create signature
            signature = hmac.new(
                self.cipher._signing_key,
                encrypted,
                hashlib.sha256
            ).hexdigest()

            return {
                "encrypted": encrypted.decode(),
                "signature": signature
            }
        except Exception as e:
            self.logger.error(f"Request securing failed: {str(e)}")
            raise

    def verify_request(self, secure_request: Dict) -> bool:
        """Verify a secure request"""
        try:
            encrypted = secure_request["encrypted"].encode()
            signature = secure_request["signature"]

            # Verify signature
            expected_signature = hmac.new(
                self.cipher._signing_key,
                encrypted,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                self.logger.warning("Request signature verification failed")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Request verification failed: {str(e)}")
            return False

    def decrypt_request(self, secure_request: Dict) -> Dict:
        """Decrypt a secure request"""
        try:
            encrypted = secure_request["encrypted"].encode()
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Request decryption failed: {str(e)}")
            raise


class Authentication:
    """Authentication and authorization"""

    def __init__(self, secret: str, algorithm: str, expiration: int):
        self.secret = secret
        self.algorithm = algorithm
        self.expiration = expiration
        self.logger = logging.getLogger(__name__)

    def generate_token(self, user_id: str, roles: List[str]) -> str:
        """Generate JWT token"""
        if not JWT_AVAILABLE:
            return "jwt_not_available"
            
        try:
            payload = {
                "sub": user_id,
                "roles": roles,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=self.expiration)
            }
            return jwt.encode(payload, self.secret, algorithm=self.algorithm)
        except Exception as e:
            self.logger.error(f"Token generation failed: {str(e)}")
            raise

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token"""
        if not JWT_AVAILABLE:
            return None
            
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
        except Exception as e:
            self.logger.error(f"Token validation failed: {str(e)}")
            return None

    def has_role(self, token: str, required_role: str) -> bool:
        """Check if token has required role"""
        payload = self.validate_token(token)
        if not payload:
            return False
        return required_role in payload.get("roles", [])


class APIKeyManager:
    """API key management"""

    def __init__(self, api_keys: Dict[str, Dict]):
        self.api_keys = api_keys
        self.logger = logging.getLogger(__name__)

    def validate(self, api_key: str) -> bool:
        """Validate API key"""
        if not api_key:
            return False

        # Check if key exists
        if api_key not in self.api_keys:
            self.logger.warning(f"Invalid API key: {api_key[:8]}...")
            return False

        # Check expiration
        key_info = self.api_keys[api_key]
        if "expiration" in key_info:
            expiration = datetime.fromisoformat(key_info["expiration"])
            if datetime.utcnow() > expiration:
                self.logger.warning(f"Expired API key: {api_key[:8]}...")
                return False

        return True

    def get_roles(self, api_key: str) -> List[str]:
        """Get roles for API key"""
        if not self.validate(api_key):
            return []
        return self.api_keys[api_key].get("roles", [])


class InputValidator:
    """Input validation to prevent injection attacks"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {
            "text": r'^[\w\s\-\.,;:!?\'\"\(\)\[\]\{\}]+$',
            "code": r'^[\w\s\-\.,;:!?\'\"\(\)\[\]\{\}\+\-\*/%<>=&|\^~]+$',
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "url": r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .-]*/?$',
            "number": r'^-?\d*\.?\d+$',
            "json": r'^\{.*\}$|^\[.*\]$'
        }

    def validate(self, input_data: str, input_type: str) -> bool:
        """Validate input against type-specific patterns"""
        if input_type not in self.patterns:
            self.logger.warning(f"Unknown input type: {input_type}")
            return False

        if not re.match(self.patterns[input_type], input_data):
            self.logger.warning(f"Invalid {input_type} input: {input_data[:50]}...")
            return False

        # Additional checks for specific types
        if input_type == "code":
            return self._validate_code(input_data)
        elif input_type == "json":
            return self._validate_json(input_data)

        return True

    def _validate_code(self, code: str) -> bool:
        """Additional validation for code input"""
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__\(',
            r'exec\(',
            r'eval\(',
            r'open\(',
            r'write\(',
            r'system\(',
            r'rm\s+-rf',
            r';\s*',
            r'&\s*',
            r'\|\s*'
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                self.logger.warning(f"Dangerous code pattern detected: {pattern}")
                return False

        return True

    def _validate_json(self, json_str: str) -> bool:
        """Validate JSON input"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False


class AuditLogger:
    """Audit logging for security events"""

    def __init__(self, log_file: str, retention_days: int):
        self.log_file = log_file
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)

        # Configure logging
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(handler)

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_logs, daemon=True)
        self.cleanup_thread.start()

    def log(self, event_type: str, user_id: str, details: Dict):
        """Log an audit event"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "details": details
            }
            self.audit_logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {str(e)}")

    def _cleanup_old_logs(self):
        """Clean up old log entries"""
        while True:
            try:
                cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
                cutoff_timestamp = cutoff.timestamp()

                # In a real implementation, this would clean up the log file
                # For this example, we'll just log the cleanup attempt
                self.logger.info(f"Would clean up logs older than {cutoff.isoformat()}")

                time.sleep(86400)  # Run once per day
            except Exception as e:
                self.logger.error(f"Log cleanup failed: {str(e)}")
                time.sleep(3600)  # Retry after 1 hour

