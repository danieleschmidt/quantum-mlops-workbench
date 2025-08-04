"""Secure credential management for quantum cloud providers and services."""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    
logger = logging.getLogger(__name__)


@dataclass
class Credential:
    """Secure credential storage."""
    
    name: str
    provider: str
    credential_type: str  # api_key, token, aws_credentials, etc.
    encrypted_data: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    last_used: Optional[datetime] = None
    rotation_enabled: bool = False
    
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def needs_rotation(self, rotation_days: int = 90) -> bool:
        """Check if credential needs rotation."""
        if not self.rotation_enabled:
            return False
        rotation_due = self.created_at + timedelta(days=rotation_days)
        return datetime.utcnow() > rotation_due


class EncryptionProvider:
    """Encryption provider for credential data."""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize encryption provider."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for encryption")
            
        self.password = password or os.getenv('QUANTUM_SECURITY_KEY')
        if not self.password:
            raise ValueError("Encryption password required")
            
        # Derive encryption key from password
        salt = os.getenv('QUANTUM_SECURITY_SALT', 'quantum-mlops-salt').encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        self.fernet = Fernet(key)
        
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
            
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise


class SecureCredentialStore:
    """Secure storage backend for credentials."""
    
    def __init__(self, store_path: str = None, encryption_key: str = None):
        """Initialize secure credential store."""
        self.store_path = Path(store_path or os.getenv('QUANTUM_CREDENTIAL_STORE', 
                                                      '~/.quantum_mlops/credentials.enc')).expanduser()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions on credential store
        try:
            os.chmod(self.store_path.parent, 0o700)
            if self.store_path.exists():
                os.chmod(self.store_path, 0o600)
        except OSError as e:
            logger.warning(f"Could not set secure permissions: {e}")
            
        self.encryption = EncryptionProvider(encryption_key)
        self._credentials: Dict[str, Credential] = {}
        self._load_credentials()
        
    def _load_credentials(self) -> None:
        """Load credentials from encrypted storage."""
        if not self.store_path.exists():
            return
            
        try:
            with open(self.store_path, 'r') as f:
                encrypted_data = f.read()
                
            if not encrypted_data.strip():
                return
                
            decrypted_data = self.encryption.decrypt(encrypted_data)
            credentials_data = json.loads(decrypted_data)
            
            for cred_data in credentials_data:
                # Convert datetime strings back to datetime objects
                cred_data['created_at'] = datetime.fromisoformat(cred_data['created_at'])
                if cred_data.get('expires_at'):
                    cred_data['expires_at'] = datetime.fromisoformat(cred_data['expires_at'])
                if cred_data.get('last_used'):
                    cred_data['last_used'] = datetime.fromisoformat(cred_data['last_used'])
                    
                credential = Credential(**cred_data)
                self._credentials[credential.name] = credential
                
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise
            
    def _save_credentials(self) -> None:
        """Save credentials to encrypted storage."""
        try:
            credentials_data = []
            for credential in self._credentials.values():
                cred_dict = asdict(credential)
                # Convert datetime objects to ISO strings
                cred_dict['created_at'] = credential.created_at.isoformat()
                if credential.expires_at:
                    cred_dict['expires_at'] = credential.expires_at.isoformat()
                if credential.last_used:
                    cred_dict['last_used'] = credential.last_used.isoformat()
                credentials_data.append(cred_dict)
                
            json_data = json.dumps(credentials_data, indent=2)
            encrypted_data = self.encryption.encrypt(json_data)
            
            # Write atomically
            temp_path = self.store_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                f.write(encrypted_data)
            os.chmod(temp_path, 0o600)
            temp_path.replace(self.store_path)
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise
            
    def store_credential(self, credential: Credential) -> None:
        """Store a credential securely."""
        self._credentials[credential.name] = credential
        self._save_credentials()
        logger.info(f"Stored credential: {credential.name} ({credential.provider})")
        
    def get_credential(self, name: str) -> Optional[Credential]:
        """Retrieve a credential by name."""
        credential = self._credentials.get(name)
        if credential:
            # Update last used timestamp
            credential.last_used = datetime.utcnow()
            self._save_credentials()
        return credential
        
    def list_credentials(self) -> List[str]:
        """List available credential names."""
        return list(self._credentials.keys())
        
    def delete_credential(self, name: str) -> bool:
        """Delete a credential."""
        if name in self._credentials:
            del self._credentials[name]
            self._save_credentials()
            logger.info(f"Deleted credential: {name}")
            return True
        return False
        
    def rotate_credential(self, name: str, new_data: str) -> bool:
        """Rotate credential data."""
        if name not in self._credentials:
            return False
            
        credential = self._credentials[name]
        credential.encrypted_data = self.encryption.encrypt(new_data)
        credential.created_at = datetime.utcnow()
        self._save_credentials()
        logger.info(f"Rotated credential: {name}")
        return True
        
    def get_expired_credentials(self) -> List[str]:
        """Get list of expired credentials."""
        return [name for name, cred in self._credentials.items() if cred.is_expired()]
        
    def get_rotation_due_credentials(self, rotation_days: int = 90) -> List[str]:
        """Get credentials that need rotation."""
        return [name for name, cred in self._credentials.items() 
                if cred.needs_rotation(rotation_days)]


class CredentialManager:
    """Main credential management interface."""
    
    def __init__(self, store: Optional[SecureCredentialStore] = None):
        """Initialize credential manager."""
        self.store = store or SecureCredentialStore()
        self._env_prefix = "QUANTUM_"
        
    def store_aws_credentials(self, name: str, access_key: str, secret_key: str, 
                            region: str = "us-east-1", expires_days: int = None) -> None:
        """Store AWS credentials for Braket."""
        credential_data = {
            "access_key_id": access_key,
            "secret_access_key": secret_key,
            "region": region
        }
        
        encrypted_data = self.store.encryption.encrypt(json.dumps(credential_data))
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
        credential = Credential(
            name=name,
            provider="aws_braket",
            credential_type="aws_credentials",
            encrypted_data=encrypted_data,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            metadata={"region": region},
            rotation_enabled=True
        )
        
        self.store.store_credential(credential)
        
    def store_ibm_token(self, name: str, token: str, instance: str = None, 
                       expires_days: int = None) -> None:
        """Store IBM Quantum token."""
        credential_data = {
            "token": token,
            "instance": instance or "ibm-q/open/main"
        }
        
        encrypted_data = self.store.encryption.encrypt(json.dumps(credential_data))
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
        credential = Credential(
            name=name,
            provider="ibm_quantum",
            credential_type="api_token",
            encrypted_data=encrypted_data,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            metadata={"instance": instance},
            rotation_enabled=True
        )
        
        self.store.store_credential(credential)
        
    def store_ionq_credentials(self, name: str, api_key: str, expires_days: int = None) -> None:
        """Store IonQ API credentials."""
        credential_data = {"api_key": api_key}
        encrypted_data = self.store.encryption.encrypt(json.dumps(credential_data))
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
        credential = Credential(
            name=name,
            provider="ionq",
            credential_type="api_key",
            encrypted_data=encrypted_data,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rotation_enabled=True
        )
        
        self.store.store_credential(credential)
        
    def get_aws_credentials(self, name: str) -> Optional[Dict[str, str]]:
        """Get AWS credentials."""
        credential = self.store.get_credential(name)
        if not credential or credential.provider != "aws_braket":
            return None
            
        if credential.is_expired():
            logger.warning(f"AWS credential {name} is expired")
            return None
            
        try:
            decrypted_data = self.store.encryption.decrypt(credential.encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt AWS credentials: {e}")
            return None
            
    def get_ibm_credentials(self, name: str) -> Optional[Dict[str, str]]:
        """Get IBM Quantum credentials."""
        credential = self.store.get_credential(name)
        if not credential or credential.provider != "ibm_quantum":
            return None
            
        if credential.is_expired():
            logger.warning(f"IBM credential {name} is expired")
            return None
            
        try:
            decrypted_data = self.store.encryption.decrypt(credential.encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt IBM credentials: {e}")
            return None
            
    def get_ionq_credentials(self, name: str) -> Optional[Dict[str, str]]:
        """Get IonQ credentials."""
        credential = self.store.get_credential(name)
        if not credential or credential.provider != "ionq":
            return None
            
        if credential.is_expired():
            logger.warning(f"IonQ credential {name} is expired")
            return None
            
        try:
            decrypted_data = self.store.encryption.decrypt(credential.encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt IonQ credentials: {e}")
            return None
            
    def get_provider_credentials(self, provider: str) -> List[str]:
        """Get all credential names for a provider."""
        return [name for name, cred in self.store._credentials.items() 
                if cred.provider == provider]
                
    def setup_environment_credentials(self, credential_name: str) -> bool:
        """Setup environment variables from stored credentials."""
        credential = self.store.get_credential(credential_name)
        if not credential:
            return False
            
        try:
            decrypted_data = self.store.encryption.decrypt(credential.encrypted_data)
            cred_data = json.loads(decrypted_data)
            
            if credential.provider == "aws_braket":
                os.environ["AWS_ACCESS_KEY_ID"] = cred_data["access_key_id"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = cred_data["secret_access_key"]
                os.environ["AWS_DEFAULT_REGION"] = cred_data["region"]
                
            elif credential.provider == "ibm_quantum":
                os.environ["IBM_QUANTUM_TOKEN"] = cred_data["token"]
                os.environ["IBM_QUANTUM_INSTANCE"] = cred_data["instance"]
                
            elif credential.provider == "ionq":
                os.environ["IONQ_API_KEY"] = cred_data["api_key"]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup environment credentials: {e}")
            return False
            
    def validate_credentials(self, name: str) -> Dict[str, Any]:
        """Validate credential integrity and accessibility."""
        credential = self.store.get_credential(name)
        if not credential:
            return {"valid": False, "error": "Credential not found"}
            
        result = {
            "valid": True,
            "name": name,
            "provider": credential.provider,
            "type": credential.credential_type,
            "created": credential.created_at.isoformat(),
            "expired": credential.is_expired(),
            "needs_rotation": credential.needs_rotation(),
            "last_used": credential.last_used.isoformat() if credential.last_used else None
        }
        
        try:
            # Test decryption
            self.store.encryption.decrypt(credential.encrypted_data)
        except Exception as e:
            result["valid"] = False
            result["error"] = f"Decryption failed: {e}"
            
        return result
        
    def cleanup_expired_credentials(self) -> List[str]:
        """Remove expired credentials."""
        expired = self.store.get_expired_credentials()
        for name in expired:
            self.store.delete_credential(name)
        return expired
        
    def generate_rotation_report(self) -> Dict[str, Any]:
        """Generate credential rotation report."""
        all_creds = self.store.list_credentials()
        expired = self.store.get_expired_credentials()
        rotation_due = self.store.get_rotation_due_credentials()
        
        return {
            "total_credentials": len(all_creds),
            "expired_credentials": len(expired),
            "rotation_due": len(rotation_due),
            "expired_list": expired,
            "rotation_due_list": rotation_due,
            "report_generated": datetime.utcnow().isoformat()
        }


# Convenience function for global credential manager
_global_credential_manager: Optional[CredentialManager] = None

def get_credential_manager() -> CredentialManager:
    """Get global credential manager instance."""
    global _global_credential_manager
    if _global_credential_manager is None:
        _global_credential_manager = CredentialManager()
    return _global_credential_manager