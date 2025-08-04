"""Data encryption at rest and in transit for quantum MLOps workbench."""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.x509.oid import NameOID
    from cryptography import x509
    import ssl
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data."""
    
    algorithm: str
    key_id: str
    iv: Optional[str] = None
    salt: Optional[str] = None
    version: str = "1.0"
    timestamp: Optional[str] = None
    checksum: Optional[str] = None


class KeyManager:
    """Cryptographic key management."""
    
    def __init__(self, key_store_path: str = None):
        """Initialize key manager."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for encryption")
            
        self.key_store_path = Path(key_store_path or os.getenv('QUANTUM_KEY_STORE', 
                                                              '~/.quantum_mlops/keys')).expanduser()
        self.key_store_path.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions
        try:
            os.chmod(self.key_store_path, 0o700)
        except OSError as e:
            logger.warning(f"Could not set secure permissions on key store: {e}")
            
        self._keys: Dict[str, bytes] = {}
        self._load_keys()
        
    def _load_keys(self) -> None:
        """Load keys from key store."""
        key_file = self.key_store_path / "keys.enc"
        if not key_file.exists():
            return
            
        try:
            # Use master key to decrypt key store
            master_key = self._get_master_key()
            fernet = Fernet(master_key)
            
            with open(key_file, 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = fernet.decrypt(encrypted_data)
            keys_data = json.loads(decrypted_data.decode())
            
            for key_id, key_b64 in keys_data.items():
                self._keys[key_id] = base64.b64decode(key_b64)
                
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            
    def _save_keys(self) -> None:
        """Save keys to key store."""
        try:
            # Prepare keys data
            keys_data = {}
            for key_id, key_bytes in self._keys.items():
                keys_data[key_id] = base64.b64encode(key_bytes).decode()
                
            # Encrypt with master key
            master_key = self._get_master_key()
            fernet = Fernet(master_key)
            
            json_data = json.dumps(keys_data).encode()
            encrypted_data = fernet.encrypt(json_data)
            
            # Write atomically
            key_file = self.key_store_path / "keys.enc"
            temp_file = key_file.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                f.write(encrypted_data)
                
            os.chmod(temp_file, 0o600)
            temp_file.replace(key_file)
            
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
            
    def _get_master_key(self) -> bytes:
        """Get or create master encryption key."""
        master_key_file = self.key_store_path / "master.key"
        
        if master_key_file.exists():
            with open(master_key_file, 'rb') as f:
                return f.read()
        else:
            # Create new master key
            master_key = Fernet.generate_key()
            with open(master_key_file, 'wb') as f:
                f.write(master_key)
            os.chmod(master_key_file, 0o600)
            return master_key
            
    def generate_key(self, key_id: str, algorithm: str = "fernet") -> str:
        """Generate new encryption key."""
        if algorithm == "fernet":
            key = Fernet.generate_key()
        elif algorithm == "aes256":
            key = secrets.token_bytes(32)  # 256-bit key
        elif algorithm == "aes128":
            key = secrets.token_bytes(16)  # 128-bit key
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        self._keys[key_id] = key
        self._save_keys()
        logger.info(f"Generated encryption key: {key_id}")
        return key_id
        
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get encryption key by ID."""
        return self._keys.get(key_id)
        
    def delete_key(self, key_id: str) -> bool:
        """Delete encryption key."""
        if key_id in self._keys:
            del self._keys[key_id]
            self._save_keys()
            logger.info(f"Deleted encryption key: {key_id}")
            return True
        return False
        
    def list_keys(self) -> List[str]:
        """List available key IDs."""
        return list(self._keys.keys())
        
    def rotate_key(self, key_id: str, algorithm: str = "fernet") -> str:
        """Rotate encryption key."""
        if key_id not in self._keys:
            raise ValueError(f"Key {key_id} not found")
            
        # Generate new key
        old_key_id = f"{key_id}_old_{secrets.token_hex(4)}"
        self._keys[old_key_id] = self._keys[key_id]
        
        # Create new key
        return self.generate_key(key_id, algorithm)


class SymmetricEncryption:
    """Symmetric encryption for data at rest."""
    
    def __init__(self, key_manager: KeyManager):
        """Initialize symmetric encryption."""
        self.key_manager = key_manager
        
    def encrypt_data(self, data: bytes, key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt data with symmetric encryption."""
        if key_id is None:
            key_id = "default"
            
        key = self.key_manager.get_key(key_id)
        if not key:
            # Generate new key
            self.key_manager.generate_key(key_id)
            key = self.key_manager.get_key(key_id)
            
        # Use Fernet for symmetric encryption
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        
        metadata = EncryptionMetadata(
            algorithm="fernet",
            key_id=key_id,
            checksum=hashlib.sha256(data).hexdigest()
        )
        
        return encrypted_data, metadata
        
    def decrypt_data(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt data with symmetric encryption."""
        key = self.key_manager.get_key(metadata.key_id)
        if not key:
            raise ValueError(f"Encryption key {metadata.key_id} not found")
            
        if metadata.algorithm == "fernet":
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
        else:
            raise ValueError(f"Unsupported algorithm: {metadata.algorithm}")
            
        # Verify checksum if available
        if metadata.checksum:
            actual_checksum = hashlib.sha256(decrypted_data).hexdigest()
            if actual_checksum != metadata.checksum:
                raise ValueError("Data integrity check failed")
                
        return decrypted_data
        
    def encrypt_json(self, data: Dict[str, Any], key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt JSON data."""
        json_bytes = json.dumps(data, separators=(',', ':')).encode()
        return self.encrypt_data(json_bytes, key_id)
        
    def decrypt_json(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Dict[str, Any]:
        """Decrypt JSON data."""
        decrypted_bytes = self.decrypt_data(encrypted_data, metadata)
        return json.loads(decrypted_bytes.decode())


class QuantumDataEncryption:
    """Specialized encryption for quantum data."""
    
    def __init__(self, symmetric_encryption: SymmetricEncryption):
        """Initialize quantum data encryption."""
        self.symmetric_encryption = symmetric_encryption
        
    def encrypt_quantum_circuit(self, circuit: Dict[str, Any], key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt quantum circuit data."""
        # Sanitize circuit before encryption
        sanitized_circuit = self._sanitize_circuit(circuit)
        return self.symmetric_encryption.encrypt_json(sanitized_circuit, key_id)
        
    def decrypt_quantum_circuit(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Dict[str, Any]:
        """Decrypt quantum circuit data."""
        return self.symmetric_encryption.decrypt_json(encrypted_data, metadata)
        
    def encrypt_quantum_parameters(self, parameters: Union[List, Dict], key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt quantum parameters."""
        if NUMPY_AVAILABLE and isinstance(parameters, np.ndarray):
            # Convert numpy array to list for JSON serialization
            param_data = {
                "type": "numpy_array",
                "shape": parameters.shape,
                "dtype": str(parameters.dtype),
                "data": parameters.tolist()
            }
        else:
            param_data = {
                "type": "native",
                "data": parameters
            }
            
        return self.symmetric_encryption.encrypt_json(param_data, key_id)
        
    def decrypt_quantum_parameters(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Union[List, Dict]:
        """Decrypt quantum parameters."""
        param_data = self.symmetric_encryption.decrypt_json(encrypted_data, metadata)
        
        if param_data.get("type") == "numpy_array" and NUMPY_AVAILABLE:
            return np.array(param_data["data"], dtype=param_data["dtype"]).reshape(param_data["shape"])
        else:
            return param_data["data"]
            
    def encrypt_quantum_results(self, results: Dict[str, Any], key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt quantum execution results."""
        # Clean results data
        clean_results = self._sanitize_results(results)
        return self.symmetric_encryption.encrypt_json(clean_results, key_id)
        
    def decrypt_quantum_results(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Dict[str, Any]:
        """Decrypt quantum execution results."""
        return self.symmetric_encryption.decrypt_json(encrypted_data, metadata)
        
    def _sanitize_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize circuit data before encryption."""
        sanitized = {}
        
        # Only include safe fields
        safe_fields = ['n_qubits', 'gates', 'measurements', 'parameters', 'metadata']
        for field in safe_fields:
            if field in circuit:
                sanitized[field] = circuit[field]
                
        # Ensure gates are properly formatted
        if 'gates' in sanitized:
            sanitized_gates = []
            for gate in sanitized['gates']:
                if isinstance(gate, dict):
                    # Only include safe gate fields
                    safe_gate_fields = ['type', 'wires', 'qubit', 'qubits', 'angle', 'angles', 'parameters']
                    sanitized_gate = {field: gate[field] for field in safe_gate_fields if field in gate}
                    sanitized_gates.append(sanitized_gate)
            sanitized['gates'] = sanitized_gates
            
        return sanitized
        
    def _sanitize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize results data before encryption."""
        sanitized = {}
        
        # Only include safe fields
        safe_fields = ['counts', 'measurements', 'expectation_values', 'probabilities', 
                      'success', 'job_id', 'execution_time', 'shots']
        for field in safe_fields:
            if field in results:
                sanitized[field] = results[field]
                
        return sanitized


class TransitEncryption:
    """Encryption for data in transit."""
    
    def __init__(self, cert_path: str = None, key_path: str = None):
        """Initialize transit encryption."""
        self.cert_path = cert_path
        self.key_path = key_path
        
    def create_ssl_context(self, server_side: bool = False) -> ssl.SSLContext:
        """Create SSL context for secure communication."""
        if server_side:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            if self.cert_path and self.key_path:
                context.load_cert_chain(self.cert_path, self.key_path)
        else:
            context = ssl.create_default_context()
            context.check_hostname = False  # For development/testing
            context.verify_mode = ssl.CERT_NONE  # For development/testing
            
        # Set strong ciphers
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        return context
        
    def generate_self_signed_cert(self, hostname: str = "localhost", 
                                 cert_file: str = None, key_file: str = None) -> Tuple[str, str]:
        """Generate self-signed certificate for development."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for certificate generation")
            
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Quantum MLOps"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        cert_path = cert_file or "quantum_mlops.crt"
        key_path = key_file or "quantum_mlops.key"
        
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            
        # Set secure permissions
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
        
        return cert_path, key_path


class EncryptionManager:
    """Main encryption manager for quantum MLOps."""
    
    def __init__(self, key_manager: KeyManager = None):
        """Initialize encryption manager."""
        self.key_manager = key_manager or KeyManager()
        self.symmetric_encryption = SymmetricEncryption(self.key_manager)
        self.quantum_encryption = QuantumDataEncryption(self.symmetric_encryption)
        self.transit_encryption = TransitEncryption()
        
    def encrypt_quantum_data(self, data_type: str, data: Any, key_id: str = None) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt quantum data based on type."""
        if data_type == "circuit":
            return self.quantum_encryption.encrypt_quantum_circuit(data, key_id)
        elif data_type == "parameters":
            return self.quantum_encryption.encrypt_quantum_parameters(data, key_id)
        elif data_type == "results":
            return self.quantum_encryption.encrypt_quantum_results(data, key_id)
        elif data_type == "json":
            return self.symmetric_encryption.encrypt_json(data, key_id)
        else:
            # Default to raw data encryption
            if isinstance(data, str):
                data = data.encode()
            elif not isinstance(data, bytes):
                data = json.dumps(data).encode()
            return self.symmetric_encryption.encrypt_data(data, key_id)
            
    def decrypt_quantum_data(self, data_type: str, encrypted_data: bytes, 
                           metadata: EncryptionMetadata) -> Any:
        """Decrypt quantum data based on type."""
        if data_type == "circuit":
            return self.quantum_encryption.decrypt_quantum_circuit(encrypted_data, metadata)
        elif data_type == "parameters":
            return self.quantum_encryption.decrypt_quantum_parameters(encrypted_data, metadata)
        elif data_type == "results":
            return self.quantum_encryption.decrypt_quantum_results(encrypted_data, metadata)
        elif data_type == "json":
            return self.symmetric_encryption.decrypt_json(encrypted_data, metadata)
        else:
            # Default to raw data decryption
            return self.symmetric_encryption.decrypt_data(encrypted_data, metadata)
            
    def setup_secure_communication(self, hostname: str = "localhost") -> Dict[str, str]:
        """Setup secure communication certificates."""
        cert_path, key_path = self.transit_encryption.generate_self_signed_cert(hostname)
        self.transit_encryption.cert_path = cert_path
        self.transit_encryption.key_path = key_path
        
        return {
            "cert_path": cert_path,
            "key_path": key_path,
            "ssl_context": "configured"
        }
        
    def get_ssl_context(self, server_side: bool = False) -> ssl.SSLContext:
        """Get SSL context for secure communication."""
        return self.transit_encryption.create_ssl_context(server_side)
        
    def generate_encryption_key(self, key_id: str, algorithm: str = "fernet") -> str:
        """Generate new encryption key."""
        return self.key_manager.generate_key(key_id, algorithm)
        
    def rotate_encryption_key(self, key_id: str) -> str:
        """Rotate encryption key."""
        return self.key_manager.rotate_key(key_id)
        
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get encryption system status."""
        return {
            "available_keys": len(self.key_manager.list_keys()),
            "algorithms_supported": ["fernet", "aes256", "aes128"],
            "crypto_available": CRYPTO_AVAILABLE,
            "transit_encryption": {
                "cert_configured": bool(self.transit_encryption.cert_path),
                "key_configured": bool(self.transit_encryption.key_path)
            }
        }


class DataEncryption:
    """High-level interface for data encryption."""
    
    def __init__(self, encryption_manager: EncryptionManager = None):
        """Initialize data encryption."""
        self.encryption_manager = encryption_manager or EncryptionManager()
        
    def encrypt_model(self, model_data: Dict[str, Any], model_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt quantum model data."""
        key_id = f"model_{model_id}"
        return self.encryption_manager.encrypt_quantum_data("json", model_data, key_id)
        
    def decrypt_model(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Dict[str, Any]:
        """Decrypt quantum model data."""
        return self.encryption_manager.decrypt_quantum_data("json", encrypted_data, metadata)
        
    def encrypt_experiment(self, experiment_data: Dict[str, Any], experiment_id: str) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt experiment data."""
        key_id = f"experiment_{experiment_id}"
        return self.encryption_manager.encrypt_quantum_data("json", experiment_data, key_id)
        
    def decrypt_experiment(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> Dict[str, Any]:
        """Decrypt experiment data."""
        return self.encryption_manager.decrypt_quantum_data("json", encrypted_data, metadata)


# Global encryption manager
_global_encryption_manager: Optional[EncryptionManager] = None

def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager."""
    global _global_encryption_manager
    if _global_encryption_manager is None:
        _global_encryption_manager = EncryptionManager()
    return _global_encryption_manager