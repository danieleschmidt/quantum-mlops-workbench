# Quantum Cloud Provider Security Best Practices

## Overview

Quantum cloud providers offer access to real quantum hardware and advanced simulators, but they also introduce unique security considerations. This document provides comprehensive security guidance for working with major quantum cloud platforms including IBM Quantum, AWS Braket, IonQ, Rigetti, and others.

## 🎯 Quantum Cloud Security Objectives

### Core Security Principles
- **Credential Protection**: Secure management of quantum cloud provider credentials
- **Data Confidentiality**: Protect sensitive quantum algorithms and training data
- **Hardware Isolation**: Ensure proper quantum job isolation and resource access
- **Cost Management**: Prevent unauthorized usage and cost overruns
- **Compliance**: Meet regulatory requirements for cloud quantum computing

### Quantum-Specific Risks
- **Algorithm Theft**: Proprietary quantum algorithms exposed through cloud APIs
- **Hardware Fingerprinting**: Quantum device characteristics revealing sensitive information
- **Queue Timing Attacks**: Information leakage through job queue analysis
- **Measurement Data Exposure**: Quantum measurement results containing sensitive patterns
- **Cross-Talk Interference**: Adjacent quantum jobs affecting security-sensitive computations

## 🔐 Credential Management

### 1. IBM Quantum Network Security

#### Secure Token Management
```python
#!/usr/bin/env python3
"""
Secure IBM Quantum token management
"""
import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from qiskit import IBMQ

class SecureIBMQuantumCredentials:
    def __init__(self, credentials_file: str = None):
        self.credentials_file = credentials_file or os.path.expanduser('~/.qiskit/encrypted_credentials.json')
        self.key_file = os.path.expanduser('~/.qiskit/credential_key.key')
        
    def generate_key(self) -> bytes:
        """Generate encryption key for credentials"""
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        with open(self.key_file, 'wb') as f:
            f.write(key)
        
        # Set secure file permissions (Unix-like systems)
        os.chmod(self.key_file, 0o600)
        return key
    
    def load_key(self) -> bytes:
        """Load encryption key"""
        if not os.path.exists(self.key_file):
            return self.generate_key()
        
        with open(self.key_file, 'rb') as f:
            return f.read()
    
    def encrypt_credentials(self, token: str, hub: str = None, group: str = None, project: str = None):
        """Encrypt IBM Quantum credentials"""
        key = self.load_key()
        fernet = Fernet(key)
        
        credentials = {
            'token': token,
            'hub': hub,
            'group': group,
            'project': project
        }
        
        encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
        
        os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
        with open(self.credentials_file, 'wb') as f:
            f.write(encrypted_data)
        
        os.chmod(self.credentials_file, 0o600)
        print("Credentials encrypted and stored securely")
    
    def decrypt_credentials(self) -> dict:
        """Decrypt IBM Quantum credentials"""
        if not os.path.exists(self.credentials_file):
            raise FileNotFoundError("Encrypted credentials file not found")
        
        key = self.load_key()
        fernet = Fernet(key)
        
        with open(self.credentials_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def authenticate(self):
        """Securely authenticate with IBM Quantum"""
        try:
            credentials = self.decrypt_credentials()
            
            if credentials['hub']:
                IBMQ.save_account(
                    token=credentials['token'],
                    hub=credentials['hub'],
                    group=credentials['group'],
                    project=credentials['project'],
                    overwrite=True
                )
            else:
                IBMQ.save_account(
                    token=credentials['token'],
                    overwrite=True
                )
            
            IBMQ.load_account()
            print("Successfully authenticated with IBM Quantum")
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            raise

# Usage example
if __name__ == "__main__":
    # Setup credentials (run once)
    creds = SecureIBMQuantumCredentials()
    # creds.encrypt_credentials("your_ibm_quantum_token", "hub-name", "group-name", "project-name")
    
    # Use in application
    creds.authenticate()
```

#### Environment-based Configuration
```bash
# .env.example - Environment variables for IBM Quantum
# Never commit this file with real values!

# IBM Quantum credentials
IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_HUB=your_hub
IBM_QUANTUM_GROUP=your_group  
IBM_QUANTUM_PROJECT=your_project

# Security settings
IBM_QUANTUM_VERIFY_SSL=true
IBM_QUANTUM_TIMEOUT=300
IBM_QUANTUM_MAX_RETRIES=3
```

```python
# secure_ibm_config.py
import os
from qiskit import IBMQ
from dotenv import load_dotenv

class SecureIBMConfig:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('IBM_QUANTUM_TOKEN')
        self.hub = os.getenv('IBM_QUANTUM_HUB')
        self.group = os.getenv('IBM_QUANTUM_GROUP')
        self.project = os.getenv('IBM_QUANTUM_PROJECT')
        
        if not self.token:
            raise ValueError("IBM_QUANTUM_TOKEN environment variable not set")
    
    def get_provider(self):
        """Get IBM Quantum provider with security settings"""
        if not IBMQ.active_account():
            IBMQ.save_account(
                token=self.token,
                hub=self.hub,
                group=self.group,
                project=self.project,
                overwrite=True
            )
            IBMQ.load_account()
        
        return IBMQ.get_provider(
            hub=self.hub,
            group=self.group,
            project=self.project
        )
```

### 2. AWS Braket Security

#### IAM Role Configuration
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BraketQuantumJobAccess",
      "Effect": "Allow",
      "Action": [
        "braket:CreateQuantumTask",
        "braket:GetQuantumTask",
        "braket:CancelQuantumTask",
        "braket:SearchQuantumTasks"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "braket:deviceType": ["SIMULATOR", "QPU"]
        },
        "NumericLessThan": {
          "braket:maxShots": "10000"
        }
      }
    },
    {
      "Sid": "BraketDeviceAccess",
      "Effect": "Allow",
      "Action": [
        "braket:GetDevice",
        "braket:SearchDevices"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::quantum-mlops-bucket/*"
      ]
    },
    {
      "Sid": "CloudWatchLogging",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/braket/*"
    }
  ]
}
```

#### Secure Braket Configuration
```python
#!/usr/bin/env python3
"""
Secure AWS Braket configuration and usage
"""
import boto3
import json
import os
from typing import Dict, List, Optional
from botocore.exceptions import ClientError
from braket.aws import AwsDevice
from braket.devices import LocalSimulator

class SecureBraketConfig:
    def __init__(self, 
                 region: str = 'us-east-1',
                 s3_bucket: str = None,
                 role_arn: str = None):
        self.region = region
        self.s3_bucket = s3_bucket or os.getenv('BRAKET_S3_BUCKET')
        self.role_arn = role_arn or os.getenv('BRAKET_EXECUTION_ROLE_ARN')
        self.session = self._create_secure_session()
        self.braket_client = self.session.client('braket', region_name=region)
    
    def _create_secure_session(self) -> boto3.Session:
        """Create secure AWS session with proper credential handling"""
        # Use IAM roles when running in AWS environment
        if os.getenv('AWS_EXECUTION_ENV') or os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
            return boto3.Session()
        
        # Use environment variables for local development
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        session_token = os.getenv('AWS_SESSION_TOKEN')
        
        if not access_key or not secret_key:
            raise ValueError("AWS credentials not properly configured")
        
        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=self.region
        )
    
    def validate_device_access(self, device_arn: str) -> bool:
        """Validate access to specific quantum device"""
        try:
            response = self.braket_client.get_device(deviceArn=device_arn)
            device_status = response['deviceStatus']
            
            if device_status != 'ONLINE':
                print(f"Warning: Device {device_arn} is {device_status}")
                return False
            
            return True
            
        except ClientError as e:
            print(f"Access denied for device {device_arn}: {e}")
            return False
    
    def get_secure_device(self, device_arn: str) -> Optional[AwsDevice]:
        """Get quantum device with security validation"""
        if not self.validate_device_access(device_arn):
            return None
        
        try:
            device = AwsDevice(
                device_arn,
                aws_session=self.session
            )
            
            # Log device access for audit trail
            self._log_device_access(device_arn)
            
            return device
            
        except Exception as e:
            print(f"Failed to access device {device_arn}: {e}")
            return None
    
    def _log_device_access(self, device_arn: str):
        """Log device access for security auditing"""
        audit_log = {
            'timestamp': boto3.utcnow().isoformat(),
            'device_arn': device_arn,
            'user': os.getenv('USER', 'unknown'),
            'region': self.region
        }
        
        # Send to CloudWatch Logs for centralized logging
        try:
            logs_client = self.session.client('logs', region_name=self.region)
            logs_client.put_log_events(
                logGroupName='/aws/braket/security-audit',
                logStreamName=f"device-access-{device_arn.split('/')[-1]}",
                logEvents=[
                    {
                        'timestamp': int(boto3.utcnow().timestamp() * 1000),
                        'message': json.dumps(audit_log)
                    }
                ]
            )
        except Exception as e:
            print(f"Failed to log device access: {e}")
    
    def setup_cost_controls(self, max_cost_per_day: float = 100.0):
        """Set up cost controls and budgets"""
        budgets_client = self.session.client('budgets')
        
        budget_definition = {
            'BudgetName': 'QuantumMLOpsDailyBudget',
            'BudgetLimit': {
                'Amount': str(max_cost_per_day),
                'Unit': 'USD'
            },
            'TimeUnit': 'DAILY',
            'BudgetType': 'COST',
            'CostFilters': {
                'Service': ['Amazon Braket']
            }
        }
        
        try:
            budgets_client.create_budget(
                AccountId=boto3.client('sts').get_caller_identity()['Account'],
                Budget=budget_definition
            )
            print(f"Daily budget of ${max_cost_per_day} configured")
        except ClientError as e:
            if e.response['Error']['Code'] != 'DuplicateRecordException':
                print(f"Failed to create budget: {e}")

# Usage example
config = SecureBraketConfig()
device = config.get_secure_device('arn:aws:braket:::device/quantum-simulator/amazon/sv1')
config.setup_cost_controls(max_cost_per_day=50.0)
```

### 3. Multi-Provider Credential Management

#### Unified Credential Manager
```python
#!/usr/bin/env python3
"""
Unified quantum cloud provider credential management
"""
import os
import json
import keyring
from typing import Dict, Any, Optional
from enum import Enum
from cryptography.fernet import Fernet

class QuantumProvider(Enum):
    IBM_QUANTUM = "ibm_quantum"
    AWS_BRAKET = "aws_braket"
    IONQ = "ionq"
    RIGETTI = "rigetti"
    XANADU = "xanadu"

class QuantumCredentialManager:
    def __init__(self, use_system_keyring: bool = True):
        self.use_system_keyring = use_system_keyring
        self.service_name = "quantum-mlops-credentials"
        
    def store_credentials(self, 
                         provider: QuantumProvider, 
                         credentials: Dict[str, Any],
                         username: str = "default"):
        """Store encrypted credentials for quantum provider"""
        
        if self.use_system_keyring:
            self._store_in_keyring(provider, credentials, username)
        else:
            self._store_in_file(provider, credentials, username)
    
    def retrieve_credentials(self, 
                           provider: QuantumProvider, 
                           username: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve decrypted credentials for quantum provider"""
        
        if self.use_system_keyring:
            return self._retrieve_from_keyring(provider, username)
        else:
            return self._retrieve_from_file(provider, username)
    
    def _store_in_keyring(self, 
                         provider: QuantumProvider, 
                         credentials: Dict[str, Any],
                         username: str):
        """Store credentials using system keyring"""
        keyring.set_password(
            self.service_name,
            f"{provider.value}_{username}",
            json.dumps(credentials)
        )
    
    def _retrieve_from_keyring(self, 
                              provider: QuantumProvider, 
                              username: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials from system keyring"""
        try:
            credentials_json = keyring.get_password(
                self.service_name,
                f"{provider.value}_{username}"
            )
            
            if credentials_json:
                return json.loads(credentials_json)
            return None
            
        except Exception as e:
            print(f"Failed to retrieve credentials: {e}")
            return None
    
    def _store_in_file(self, 
                      provider: QuantumProvider, 
                      credentials: Dict[str, Any],
                      username: str):
        """Store credentials in encrypted file"""
        credentials_dir = os.path.expanduser('~/.quantum-mlops/credentials')
        os.makedirs(credentials_dir, exist_ok=True)
        
        # Generate or load encryption key
        key_file = os.path.join(credentials_dir, 'key.key')
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
        
        # Encrypt and store credentials
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
        
        credentials_file = os.path.join(
            credentials_dir, 
            f"{provider.value}_{username}.enc"
        )
        
        with open(credentials_file, 'wb') as f:
            f.write(encrypted_data)
        
        os.chmod(credentials_file, 0o600)
    
    def _retrieve_from_file(self, 
                           provider: QuantumProvider, 
                           username: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials from encrypted file"""
        credentials_dir = os.path.expanduser('~/.quantum-mlops/credentials')
        
        key_file = os.path.join(credentials_dir, 'key.key')
        credentials_file = os.path.join(
            credentials_dir, 
            f"{provider.value}_{username}.enc"
        )
        
        if not os.path.exists(key_file) or not os.path.exists(credentials_file):
            return None
        
        try:
            with open(key_file, 'rb') as f:
                key = f.read()
            
            with open(credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return json.loads(decrypted_data.decode())
            
        except Exception as e:
            print(f"Failed to decrypt credentials: {e}")
            return None
    
    def delete_credentials(self, 
                          provider: QuantumProvider, 
                          username: str = "default"):
        """Delete stored credentials"""
        if self.use_system_keyring:
            try:
                keyring.delete_password(
                    self.service_name,
                    f"{provider.value}_{username}"
                )
            except keyring.errors.PasswordDeleteError:
                pass
        else:
            credentials_dir = os.path.expanduser('~/.quantum-mlops/credentials')
            credentials_file = os.path.join(
                credentials_dir, 
                f"{provider.value}_{username}.enc"
            )
            
            if os.path.exists(credentials_file):
                os.remove(credentials_file)
    
    def list_stored_providers(self) -> List[str]:
        """List all providers with stored credentials"""
        if self.use_system_keyring:
            # System keyring doesn't provide easy enumeration
            return ["Use retrieve_credentials to check individual providers"]
        else:
            credentials_dir = os.path.expanduser('~/.quantum-mlops/credentials')
            if not os.path.exists(credentials_dir):
                return []
            
            stored_providers = []
            for filename in os.listdir(credentials_dir):
                if filename.endswith('.enc'):
                    provider_username = filename.replace('.enc', '')
                    stored_providers.append(provider_username)
            
            return stored_providers

# Example usage and setup
def setup_quantum_credentials():
    """Setup script for quantum provider credentials"""
    manager = QuantumCredentialManager()
    
    # IBM Quantum
    ibm_creds = {
        'token': input("Enter IBM Quantum token: "),
        'hub': input("Enter IBM Quantum hub (optional): ") or None,
        'group': input("Enter IBM Quantum group (optional): ") or None,
        'project': input("Enter IBM Quantum project (optional): ") or None
    }
    manager.store_credentials(QuantumProvider.IBM_QUANTUM, ibm_creds)
    
    # AWS Braket
    aws_creds = {
        'access_key_id': input("Enter AWS Access Key ID: "),
        'secret_access_key': input("Enter AWS Secret Access Key: "),
        'region': input("Enter AWS region (default: us-east-1): ") or "us-east-1",
        's3_bucket': input("Enter S3 bucket for Braket results: ")
    }
    manager.store_credentials(QuantumProvider.AWS_BRAKET, aws_creds)
    
    # IonQ
    ionq_creds = {
        'api_key': input("Enter IonQ API key: "),
        'base_url': input("Enter IonQ base URL (optional): ") or None
    }
    manager.store_credentials(QuantumProvider.IONQ, ionq_creds)
    
    print("Credentials stored securely!")

if __name__ == "__main__":
    setup_quantum_credentials()
```

## 🛡️ Data Protection and Privacy

### 1. Quantum Algorithm Protection

#### Circuit Obfuscation
```python
#!/usr/bin/env python3
"""
Quantum circuit obfuscation for algorithm protection
"""
import numpy as np
from typing import List, Tuple, Dict
import pennylane as qml
from pennylane import numpy as pnp

class QuantumCircuitObfuscator:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.obfuscation_key = self._generate_obfuscation_key()
    
    def _generate_obfuscation_key(self) -> Dict[str, np.ndarray]:
        """Generate random obfuscation parameters"""
        np.random.seed(42)  # Use secure random in production
        
        return {
            'rotation_angles': np.random.uniform(0, 2*np.pi, self.n_qubits),
            'permutation_map': np.random.permutation(self.n_qubits),
            'phase_shifts': np.random.uniform(0, 2*np.pi, self.n_qubits)
        }
    
    def obfuscate_circuit(self, circuit_func):
        """Obfuscate quantum circuit to protect IP"""
        def obfuscated_circuit(params, x):
            # Apply initial obfuscation rotations
            for i in range(self.n_qubits):
                qml.RY(self.obfuscation_key['rotation_angles'][i], wires=i)
            
            # Apply permutation to qubit ordering
            original_qubits = list(range(self.n_qubits))
            permuted_qubits = [original_qubits[self.obfuscation_key['permutation_map'][i]] 
                             for i in range(self.n_qubits)]
            
            # Execute original circuit with permuted qubits
            # Note: This is a simplified example - full implementation would
            # need to handle the permutation more carefully
            result = circuit_func(params, x)
            
            # Apply phase obfuscation
            for i in range(self.n_qubits):
                qml.PhaseShift(self.obfuscation_key['phase_shifts'][i], wires=i)
            
            return result
        
        return obfuscated_circuit
    
    def create_secure_device(self, provider: str = "default.qubit"):
        """Create quantum device with security monitoring"""
        dev = qml.device(provider, wires=self.n_qubits)
        
        # Wrap device to add security logging
        original_execute = dev.execute
        
        def secure_execute(circuits, execution_config=None):
            # Log circuit execution for security audit
            self._log_circuit_execution(circuits)
            
            # Check for suspicious patterns
            if self._detect_suspicious_access(circuits):
                raise SecurityError("Suspicious circuit access pattern detected")
            
            return original_execute(circuits, execution_config)
        
        dev.execute = secure_execute
        return dev
    
    def _log_circuit_execution(self, circuits):
        """Log circuit execution for security monitoring"""
        import datetime
        
        log_entry = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'circuit_count': len(circuits) if isinstance(circuits, list) else 1,
            'qubit_count': self.n_qubits,
            'obfuscated': True
        }
        
        # In production, send to secure logging service
        print(f"Circuit execution logged: {log_entry}")
    
    def _detect_suspicious_access(self, circuits) -> bool:
        """Detect suspicious circuit access patterns"""
        # Implement pattern detection logic
        # This is a simplified example
        
        if isinstance(circuits, list) and len(circuits) > 100:
            print("Warning: High-frequency circuit execution detected")
            return True
        
        return False

class SecurityError(Exception):
    pass

# Example usage
obfuscator = QuantumCircuitObfuscator(n_qubits=4)

@qml.qnode(obfuscator.create_secure_device())
def secure_quantum_model(params, x):
    # Your proprietary quantum algorithm here
    qml.templates.AngleEmbedding(x, wires=range(4))
    qml.templates.BasicEntanglerLayers(params, wires=range(4))
    return qml.expval(qml.PauliZ(0))

# Obfuscate the circuit
protected_model = obfuscator.obfuscate_circuit(secure_quantum_model)
```

### 2. Secure Data Handling

#### Data Encryption for Quantum ML
```python
#!/usr/bin/env python3
"""
Secure data handling for quantum machine learning
"""
import numpy as np
from cryptography.fernet import Fernet
from typing import Union, Tuple
import json
import base64

class QuantumDataSecurityManager:
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_training_data(self, 
                            X: np.ndarray, 
                            y: np.ndarray = None) -> Tuple[bytes, bytes]:
        """Encrypt quantum training data"""
        
        # Serialize numpy arrays
        X_serialized = json.dumps({
            'data': X.tolist(),
            'shape': X.shape,
            'dtype': str(X.dtype)
        })
        
        y_serialized = None
        if y is not None:
            y_serialized = json.dumps({
                'data': y.tolist(),
                'shape': y.shape,
                'dtype': str(y.dtype)
            })
        
        # Encrypt serialized data
        X_encrypted = self.cipher.encrypt(X_serialized.encode())
        y_encrypted = self.cipher.encrypt(y_serialized.encode()) if y_serialized else None
        
        return X_encrypted, y_encrypted
    
    def decrypt_training_data(self, 
                            X_encrypted: bytes, 
                            y_encrypted: bytes = None) -> Tuple[np.ndarray, np.ndarray]:
        """Decrypt quantum training data"""
        
        # Decrypt data
        X_serialized = self.cipher.decrypt(X_encrypted).decode()
        X_data = json.loads(X_serialized)
        X = np.array(X_data['data'], dtype=X_data['dtype']).reshape(X_data['shape'])
        
        y = None
        if y_encrypted:
            y_serialized = self.cipher.decrypt(y_encrypted).decode()
            y_data = json.loads(y_serialized)
            y = np.array(y_data['data'], dtype=y_data['dtype']).reshape(y_data['shape'])
        
        return X, y
    
    def secure_parameter_storage(self, params: np.ndarray, metadata: dict = None) -> str:
        """Securely store quantum model parameters"""
        
        param_data = {
            'parameters': params.tolist(),
            'shape': params.shape,
            'dtype': str(params.dtype),
            'metadata': metadata or {}
        }
        
        # Encrypt parameter data
        encrypted_params = self.cipher.encrypt(json.dumps(param_data).encode())
        
        # Return base64 encoded encrypted data for storage
        return base64.b64encode(encrypted_params).decode()
    
    def load_secure_parameters(self, encrypted_data: str) -> Tuple[np.ndarray, dict]:
        """Load securely stored quantum model parameters"""
        
        # Decode and decrypt
        encrypted_params = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_params).decode()
        param_data = json.loads(decrypted_data)
        
        # Reconstruct parameters
        params = np.array(
            param_data['parameters'], 
            dtype=param_data['dtype']
        ).reshape(param_data['shape'])
        
        metadata = param_data.get('metadata', {})
        
        return params, metadata
    
    def sanitize_circuit_parameters(self, params: np.ndarray) -> np.ndarray:
        """Sanitize circuit parameters to prevent information leakage"""
        
        # Add controlled noise to parameters to prevent reverse engineering
        noise_level = 1e-10  # Very small noise that doesn't affect functionality
        noise = np.random.normal(0, noise_level, params.shape)
        
        # Clip parameters to valid ranges
        sanitized_params = np.clip(params + noise, -2*np.pi, 2*np.pi)
        
        return sanitized_params
    
    def create_secure_data_loader(self, 
                                 encrypted_X: bytes, 
                                 encrypted_y: bytes = None,
                                 batch_size: int = 32):
        """Create secure data loader for encrypted training data"""
        
        X, y = self.decrypt_training_data(encrypted_X, encrypted_y)
        
        class SecureDataLoader:
            def __init__(self, X, y, batch_size, security_manager):
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.security_manager = security_manager
                self.current_idx = 0
            
            def __iter__(self):
                self.current_idx = 0
                return self
            
            def __next__(self):
                if self.current_idx >= len(self.X):
                    raise StopIteration
                
                end_idx = min(self.current_idx + self.batch_size, len(self.X))
                
                batch_X = self.X[self.current_idx:end_idx]
                batch_y = self.y[self.current_idx:end_idx] if self.y is not None else None
                
                # Sanitize batch data
                batch_X = self.security_manager._sanitize_batch_data(batch_X)
                
                self.current_idx = end_idx
                
                return batch_X, batch_y
        
        return SecureDataLoader(X, y, batch_size, self)
    
    def _sanitize_batch_data(self, batch_data: np.ndarray) -> np.ndarray:
        """Sanitize batch data to prevent information leakage"""
        
        # Normalize data to prevent scale-based information leakage
        if batch_data.std() > 0:
            batch_data = (batch_data - batch_data.mean()) / batch_data.std()
        
        return batch_data
    
    def export_key(self) -> str:
        """Export encryption key for secure backup"""
        return base64.b64encode(self.encryption_key).decode()
    
    @classmethod
    def from_key(cls, key_string: str):
        """Create instance from exported key"""
        key = base64.b64decode(key_string.encode())
        return cls(encryption_key=key)

# Example usage
security_manager = QuantumDataSecurityManager()

# Encrypt training data
X_train = np.random.rand(100, 4)
y_train = np.random.randint(0, 2, 100)

X_encrypted, y_encrypted = security_manager.encrypt_training_data(X_train, y_train)

# Create secure data loader
data_loader = security_manager.create_secure_data_loader(
    X_encrypted, y_encrypted, batch_size=16
)

# Use in training loop
for batch_X, batch_y in data_loader:
    # Your quantum ML training code here
    print(f"Processing batch: {batch_X.shape}")
```

## 🔐 Access Control and Monitoring

### 1. Quantum Resource Access Control

#### Role-Based Access Control (RBAC)
```python
#!/usr/bin/env python3
"""
Role-based access control for quantum resources
"""
from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass
import datetime
import json

class QuantumRole(Enum):
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    ADMIN = "admin"
    AUDITOR = "auditor"

class QuantumResource(Enum):
    SIMULATOR = "simulator"
    QPU_SMALL = "qpu_small"  # < 10 qubits
    QPU_MEDIUM = "qpu_medium"  # 10-50 qubits
    QPU_LARGE = "qpu_large"  # > 50 qubits
    ALGORITHM_LIBRARY = "algorithm_library"
    TRAINING_DATA = "training_data"

@dataclass
class AccessPermission:
    resource: QuantumResource
    actions: Set[str]  # read, write, execute, delete
    constraints: Dict[str, any]  # cost limits, time limits, etc.

class QuantumAccessController:
    def __init__(self):
        self.role_permissions = self._define_role_permissions()
        self.user_roles = {}
        self.access_log = []
        self.active_sessions = {}
    
    def _define_role_permissions(self) -> Dict[QuantumRole, List[AccessPermission]]:
        """Define permissions for each role"""
        return {
            QuantumRole.RESEARCHER: [
                AccessPermission(
                    QuantumResource.SIMULATOR,
                    {"read", "execute"},
                    {"max_shots": 10000, "max_qubits": 20}
                ),
                AccessPermission(
                    QuantumResource.QPU_SMALL,
                    {"read", "execute"},
                    {"max_shots": 1000, "max_cost_per_day": 10.0}
                ),
                AccessPermission(
                    QuantumResource.ALGORITHM_LIBRARY,
                    {"read"},
                    {}
                ),
                AccessPermission(
                    QuantumResource.TRAINING_DATA,
                    {"read"},
                    {"data_classification": ["public", "internal"]}
                )
            ],
            
            QuantumRole.DEVELOPER: [
                AccessPermission(
                    QuantumResource.SIMULATOR,
                    {"read", "write", "execute"},
                    {"max_shots": 50000, "max_qubits": 25}
                ),
                AccessPermission(
                    QuantumResource.QPU_SMALL,
                    {"read", "execute"},
                    {"max_shots": 5000, "max_cost_per_day": 50.0}
                ),
                AccessPermission(
                    QuantumResource.QPU_MEDIUM,
                    {"read", "execute"},
                    {"max_shots": 1000, "max_cost_per_day": 100.0}
                ),
                AccessPermission(
                    QuantumResource.ALGORITHM_LIBRARY,
                    {"read", "write"},
                    {}
                ),
                AccessPermission(
                    QuantumResource.TRAINING_DATA,
                    {"read", "write"},
                    {"data_classification": ["public", "internal", "confidential"]}
                )
            ],
            
            QuantumRole.ADMIN: [
                AccessPermission(
                    resource,
                    {"read", "write", "execute", "delete"},
                    {}
                ) for resource in QuantumResource
            ],
            
            QuantumRole.AUDITOR: [
                AccessPermission(
                    resource,
                    {"read"},
                    {}
                ) for resource in QuantumResource
            ]
        }
    
    def assign_role(self, user_id: str, role: QuantumRole):
        """Assign role to user"""
        self.user_roles[user_id] = role
        self._log_access("role_assignment", user_id, None, f"Assigned role: {role.value}")
    
    def check_access(self, 
                    user_id: str, 
                    resource: QuantumResource, 
                    action: str,
                    context: Dict = None) -> Tuple[bool, str]:
        """Check if user has access to resource and action"""
        
        if user_id not in self.user_roles:
            return False, "User has no assigned role"
        
        user_role = self.user_roles[user_id]
        role_permissions = self.role_permissions.get(user_role, [])
        
        # Find relevant permission
        relevant_permission = None
        for permission in role_permissions:
            if permission.resource == resource:
                relevant_permission = permission
                break
        
        if not relevant_permission:
            return False, f"No permission for resource {resource.value}"
        
        if action not in relevant_permission.actions:
            return False, f"Action {action} not permitted for resource {resource.value}"
        
        # Check constraints
        if not self._check_constraints(user_id, relevant_permission.constraints, context):
            return False, "Resource constraints violated"
        
        return True, "Access granted"
    
    def _check_constraints(self, 
                          user_id: str, 
                          constraints: Dict, 
                          context: Dict = None) -> bool:
        """Check if constraints are satisfied"""
        context = context or {}
        
        # Check shot limits
        if "max_shots" in constraints:
            requested_shots = context.get("shots", 0)
            if requested_shots > constraints["max_shots"]:
                return False
        
        # Check qubit limits
        if "max_qubits" in constraints:
            requested_qubits = context.get("qubits", 0)
            if requested_qubits > constraints["max_qubits"]:
                return False
        
        # Check daily cost limits
        if "max_cost_per_day" in constraints:
            daily_cost = self._get_daily_cost(user_id)
            if daily_cost >= constraints["max_cost_per_day"]:
                return False
        
        # Check data classification
        if "data_classification" in constraints:
            data_class = context.get("data_classification", "confidential")
            if data_class not in constraints["data_classification"]:
                return False
        
        return True
    
    def _get_daily_cost(self, user_id: str) -> float:
        """Get user's cost for current day"""
        today = datetime.date.today()
        daily_cost = 0.0
        
        for log_entry in self.access_log:
            if (log_entry["user_id"] == user_id and 
                log_entry["date"] == today.isoformat() and
                "cost" in log_entry):
                daily_cost += log_entry["cost"]
        
        return daily_cost
    
    def create_secure_session(self, user_id: str, duration_hours: int = 8) -> str:
        """Create secure session for user"""
        import uuid
        
        session_id = str(uuid.uuid4())
        expiry_time = datetime.datetime.now() + datetime.timedelta(hours=duration_hours)
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.datetime.now(),
            "expires_at": expiry_time,
            "activity_count": 0
        }
        
        self._log_access("session_created", user_id, None, f"Session {session_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Tuple[bool, str]:
        """Validate active session"""
        if session_id not in self.active_sessions:
            return False, "Invalid session"
        
        session = self.active_sessions[session_id]
        if datetime.datetime.now() > session["expires_at"]:
            del self.active_sessions[session_id]
            return False, "Session expired"
        
        # Update activity
        session["activity_count"] += 1
        return True, session["user_id"]
    
    def execute_with_access_control(self, 
                                   session_id: str,
                                   resource: QuantumResource,
                                   action: str,
                                   context: Dict = None):
        """Execute action with access control"""
        
        # Validate session
        is_valid, user_id = self.validate_session(session_id)
        if not is_valid:
            raise PermissionError(f"Invalid session: {user_id}")
        
        # Check access
        has_access, message = self.check_access(user_id, resource, action, context)
        if not has_access:
            self._log_access("access_denied", user_id, resource, message)
            raise PermissionError(f"Access denied: {message}")
        
        # Log successful access
        self._log_access("access_granted", user_id, resource, f"Action: {action}")
        
        # Execute action (implement actual execution logic)
        return self._execute_quantum_action(resource, action, context)
    
    def _execute_quantum_action(self, 
                               resource: QuantumResource, 
                               action: str, 
                               context: Dict):
        """Execute the actual quantum action"""
        # This would contain the actual quantum computing logic
        print(f"Executing {action} on {resource.value} with context {context}")
        return {"status": "success", "resource": resource.value, "action": action}
    
    def _log_access(self, event_type: str, user_id: str, resource: QuantumResource, details: str):
        """Log access events for auditing"""
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource.value if resource else None,
            "details": details,
            "date": datetime.date.today().isoformat()
        }
        
        self.access_log.append(log_entry)
        
        # In production, send to centralized logging system
        print(f"Access log: {json.dumps(log_entry)}")
    
    def generate_audit_report(self, days: int = 30) -> Dict:
        """Generate audit report for specified period"""
        cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
        
        relevant_logs = [
            log for log in self.access_log
            if datetime.date.fromisoformat(log["date"]) >= cutoff_date
        ]
        
        # Aggregate statistics
        users = set(log["user_id"] for log in relevant_logs)
        resources = set(log["resource"] for log in relevant_logs if log["resource"])
        
        access_denied_count = len([log for log in relevant_logs if log["event_type"] == "access_denied"])
        access_granted_count = len([log for log in relevant_logs if log["event_type"] == "access_granted"])
        
        return {
            "period_days": days,
            "total_events": len(relevant_logs),
            "unique_users": len(users),
            "unique_resources": len(resources),
            "access_granted": access_granted_count,
            "access_denied": access_denied_count,
            "active_sessions": len(self.active_sessions),
            "detailed_logs": relevant_logs
        }

# Example usage
access_controller = QuantumAccessController()

# Assign roles
access_controller.assign_role("alice", QuantumRole.RESEARCHER)
access_controller.assign_role("bob", QuantumRole.DEVELOPER)
access_controller.assign_role("charlie", QuantumRole.ADMIN)

# Create session and execute actions
alice_session = access_controller.create_secure_session("alice")

try:
    result = access_controller.execute_with_access_control(
        alice_session,
        QuantumResource.SIMULATOR,
        "execute",
        {"shots": 5000, "qubits": 10}
    )
    print(f"Execution result: {result}")
except PermissionError as e:
    print(f"Permission error: {e}")

# Generate audit report
audit_report = access_controller.generate_audit_report(days=7)
print(f"Audit report: {json.dumps(audit_report, indent=2)}")
```

## 📊 Security Monitoring and Alerting

### 1. Quantum Security Monitoring Dashboard

#### Monitoring Configuration
```yaml
# monitoring/quantum-security-monitoring.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-security-alerts
data:
  prometheus-rules.yml: |
    groups:
    - name: quantum_security
      rules:
      # Credential exposure detection
      - alert: QuantumCredentialExposed
        expr: |
          increase(log_entries{
            level="ERROR",
            message=~".*quantum.*token.*|.*api.*key.*|.*credential.*"
          }[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Potential quantum credential exposure detected"
          description: "Found {{ $value }} log entries potentially exposing quantum credentials"
      
      # Unusual quantum resource usage
      - alert: UnusualQuantumResourceUsage
        expr: |
          rate(quantum_job_executions_total[5m]) > 
          (avg_over_time(rate(quantum_job_executions_total[1h])[7d:1h]) * 3)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Unusual quantum resource usage pattern"
          description: "Quantum job execution rate is {{ $value }} times higher than usual"
      
      # High cost usage
      - alert: QuantumCostThresholdExceeded
        expr: |
          sum(rate(quantum_cost_total[1h])) * 24 > 1000
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Daily quantum cost threshold exceeded"
          description: "Projected daily quantum costs: ${{ $value }}"
      
      # Failed authentication attempts
      - alert: QuantumAuthFailures
        expr: |
          rate(quantum_auth_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of quantum authentication failures"
          description: "{{ $value }} quantum authentication failures per second"
      
      # Suspicious circuit patterns
      - alert: SuspiciousCircuitPattern
        expr: |
          increase(quantum_circuits{
            pattern="suspicious"
          }[15m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Suspicious quantum circuit patterns detected"
          description: "Detected {{ $value }} potentially malicious quantum circuits"

  grafana-dashboard.json: |
    {
      "dashboard": {
        "title": "Quantum Security Dashboard",
        "panels": [
          {
            "title": "Quantum Authentication Status",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(quantum_auth_success_total[5m])",
                "legendFormat": "Success Rate"
              }
            ]
          },
          {
            "title": "Resource Usage by Provider",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(quantum_jobs_total[5m]) by (provider)",
                "legendFormat": "{{ provider }}"
              }
            ]
          },
          {
            "title": "Security Events Timeline",
            "type": "logs",
            "targets": [
              {
                "expr": "{job=\"quantum-security\"} |= \"security\""
              }
            ]
          },
          {
            "title": "Cost Analysis",
            "type": "graph",
            "targets": [
              {
                "expr": "sum(rate(quantum_cost_total[1h])) by (provider)",
                "legendFormat": "{{ provider }} ($/hour)"
              }
            ]
          }
        ]
      }
    }
```

### 2. Security Event Processing

#### Event Processing Pipeline
```python
#!/usr/bin/env python3
"""
Quantum security event processing and alerting
"""
import json
import re
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class SecurityEventType(Enum):
    CREDENTIAL_EXPOSURE = "credential_exposure"
    UNUSUAL_USAGE = "unusual_usage"
    AUTHENTICATION_FAILURE = "auth_failure"
    COST_THRESHOLD = "cost_threshold"
    SUSPICIOUS_CIRCUIT = "suspicious_circuit"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

@dataclass
class SecurityEvent:
    event_type: SecurityEventType
    severity: str  # critical, high, medium, low
    timestamp: datetime
    source: str
    message: str
    metadata: Dict[str, Any]
    user_id: str = None
    provider: str = None

class QuantumSecurityEventProcessor:
    def __init__(self):
        self.event_patterns = self._define_event_patterns()
        self.alert_thresholds = self._define_alert_thresholds()
        self.processed_events = []
        self.alert_handlers = {
            'slack': self._send_slack_alert,
            'email': self._send_email_alert,
            'webhook': self._send_webhook_alert
        }
    
    def _define_event_patterns(self) -> Dict[SecurityEventType, List[str]]:
        """Define regex patterns for security event detection"""
        return {
            SecurityEventType.CREDENTIAL_EXPOSURE: [
                r'token["\s]*[:=]["\s]*[A-Za-z0-9]{20,}',
                r'api[_-]?key["\s]*[:=]["\s]*[A-Za-z0-9]{20,}',
                r'secret["\s]*[:=]["\s]*[A-Za-z0-9]{20,}',
                r'password["\s]*[:=]["\s]*[A-Za-z0-9]{8,}'
            ],
            SecurityEventType.SUSPICIOUS_CIRCUIT: [
                r'circuit.*parameter.*extraction',
                r'reverse.*engineer.*quantum',
                r'parameter.*dump.*quantum',
                r'circuit.*analysis.*unauthorized'
            ]
        }
    
    def _define_alert_thresholds(self) -> Dict[str, Any]:
        """Define thresholds for different types of alerts"""
        return {
            'auth_failure_rate': 5,  # failures per minute
            'cost_daily_limit': 1000,  # USD
            'unusual_usage_multiplier': 3,  # times normal usage
            'credential_exposure_count': 1  # immediate alert
        }
    
    async def process_log_entry(self, log_entry: Dict[str, Any]) -> List[SecurityEvent]:
        """Process a single log entry for security events"""
        events = []
        
        # Check for credential exposure
        credential_events = self._detect_credential_exposure(log_entry)
        events.extend(credential_events)
        
        # Check for suspicious circuits
        circuit_events = self._detect_suspicious_circuits(log_entry)
        events.extend(circuit_events)
        
        # Check for authentication failures
        auth_events = self._detect_auth_failures(log_entry)
        events.extend(auth_events)
        
        # Process each detected event
        for event in events:
            await self._handle_security_event(event)
        
        return events
    
    def _detect_credential_exposure(self, log_entry: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect potential credential exposure in logs"""
        events = []
        message = log_entry.get('message', '')
        
        for pattern in self.event_patterns[SecurityEventType.CREDENTIAL_EXPOSURE]:
            if re.search(pattern, message, re.IGNORECASE):
                event = SecurityEvent(
                    event_type=SecurityEventType.CREDENTIAL_EXPOSURE,
                    severity='critical',
                    timestamp=datetime.fromisoformat(log_entry.get('timestamp', datetime.utcnow().isoformat())),
                    source=log_entry.get('source', 'unknown'),
                    message=f"Potential credential exposure detected: {pattern}",
                    metadata={
                        'original_message': message,
                        'pattern_matched': pattern,
                        'log_level': log_entry.get('level', 'unknown')
                    },
                    user_id=log_entry.get('user_id'),
                    provider=log_entry.get('provider')
                )
                events.append(event)
        
        return events
    
    def _detect_suspicious_circuits(self, log_entry: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect suspicious quantum circuit activities"""
        events = []
        message = log_entry.get('message', '')
        
        for pattern in self.event_patterns[SecurityEventType.SUSPICIOUS_CIRCUIT]:
            if re.search(pattern, message, re.IGNORECASE):
                event = SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_CIRCUIT,
                    severity='high',
                    timestamp=datetime.fromisoformat(log_entry.get('timestamp', datetime.utcnow().isoformat())),
                    source=log_entry.get('source', 'unknown'),
                    message=f"Suspicious circuit activity detected: {pattern}",
                    metadata={
                        'original_message': message,
                        'pattern_matched': pattern,
                        'circuit_id': log_entry.get('circuit_id'),
                        'qubit_count': log_entry.get('qubit_count')
                    },
                    user_id=log_entry.get('user_id'),
                    provider=log_entry.get('provider')
                )
                events.append(event)
        
        return events
    
    def _detect_auth_failures(self, log_entry: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect authentication failures"""
        events = []
        
        if (log_entry.get('level') == 'ERROR' and 
            'authentication' in log_entry.get('message', '').lower() and
            'failed' in log_entry.get('message', '').lower()):
            
            event = SecurityEvent(
                event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                severity='medium',
                timestamp=datetime.fromisoformat(log_entry.get('timestamp', datetime.utcnow().isoformat())),
                source=log_entry.get('source', 'unknown'),
                message="Authentication failure detected",
                metadata={
                    'original_message': log_entry.get('message'),
                    'ip_address': log_entry.get('ip_address'),
                    'user_agent': log_entry.get('user_agent')
                },
                user_id=log_entry.get('user_id'),
                provider=log_entry.get('provider')
            )
            events.append(event)
        
        return events
    
    async def _handle_security_event(self, event: SecurityEvent):
        """Handle a security event based on its severity"""
        self.processed_events.append(event)
        
        # Immediate alerting for critical events
        if event.severity == 'critical':
            await self._send_immediate_alert(event)
        
        # Log event for analysis
        self._log_security_event(event)
        
        # Check if event triggers additional actions
        await self._check_event_thresholds(event)
    
    async def _send_immediate_alert(self, event: SecurityEvent):
        """Send immediate alert for critical security events"""
        alert_message = {
            'title': f'🚨 Critical Quantum Security Alert',
            'message': event.message,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'source': event.source,
            'user_id': event.user_id,
            'provider': event.provider,
            'metadata': event.metadata
        }
        
        # Send to all configured alert channels
        for channel, handler in self.alert_handlers.items():
            try:
                await handler(alert_message)
            except Exception as e:
                print(f"Failed to send alert via {channel}: {e}")
    
    async def _check_event_thresholds(self, event: SecurityEvent):
        """Check if event frequency exceeds thresholds"""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=5)
        
        # Count recent events of the same type
        recent_events = [
            e for e in self.processed_events
            if (e.event_type == event.event_type and 
                e.timestamp >= window_start)
        ]
        
        # Check thresholds
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            if len(recent_events) >= self.alert_thresholds['auth_failure_rate']:
                await self._send_threshold_alert(event.event_type, len(recent_events))
    
    async def _send_threshold_alert(self, event_type: SecurityEventType, count: int):
        """Send alert when event threshold is exceeded"""
        alert_message = {
            'title': f'⚠️ Quantum Security Threshold Alert',
            'message': f'Threshold exceeded for {event_type.value}: {count} events in 5 minutes',
            'event_type': event_type.value,
            'count': count,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for channel, handler in self.alert_handlers.items():
            try:
                await handler(alert_message)
            except Exception as e:
                print(f"Failed to send threshold alert via {channel}: {e}")
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event for audit trail"""
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'severity': event.severity,
            'source': event.source,
            'message': event.message,
            'user_id': event.user_id,
            'provider': event.provider,
            'metadata': event.metadata
        }
        
        # In production, send to centralized security logging system
        print(f"Security event: {json.dumps(log_entry)}")
    
    async def _send_slack_alert(self, alert_message: Dict[str, Any]):
        """Send alert to Slack"""
        # Implementation for Slack webhook
        print(f"Slack alert: {alert_message['title']} - {alert_message['message']}")
    
    async def _send_email_alert(self, alert_message: Dict[str, Any]):
        """Send email alert"""
        # Implementation for email alerting
        print(f"Email alert: {alert_message['title']} - {alert_message['message']}")
    
    async def _send_webhook_alert(self, alert_message: Dict[str, Any]):
        """Send webhook alert"""
        # Implementation for webhook alerting
        print(f"Webhook alert: {alert_message['title']} - {alert_message['message']}")
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_events = [
            e for e in self.processed_events
            if e.timestamp >= cutoff_time
        ]
        
        # Aggregate by event type
        event_counts = {}
        for event in recent_events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Aggregate by severity
        severity_counts = {}
        for event in recent_events:
            severity = event.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'event_types': event_counts,
            'severity_breakdown': severity_counts,
            'unique_users': len(set(e.user_id for e in recent_events if e.user_id)),
            'unique_providers': len(set(e.provider for e in recent_events if e.provider)),
            'recent_critical_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.event_type.value,
                    'message': e.message,
                    'user_id': e.user_id
                }
                for e in recent_events
                if e.severity == 'critical'
            ]
        }

# Example usage
async def main():
    processor = QuantumSecurityEventProcessor()
    
    # Example log entries
    log_entries = [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'ERROR',
            'message': 'Authentication failed for user alice with token abc123def456',
            'source': 'quantum-api',
            'user_id': 'alice',
            'provider': 'ibm_quantum'
        },
        {
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'WARN',
            'message': 'Suspicious circuit parameter extraction attempt detected',
            'source': 'circuit-analyzer',
            'user_id': 'bob',
            'provider': 'aws_braket',
            'circuit_id': 'circuit_123'
        }
    ]
    
    # Process log entries
    for log_entry in log_entries:
        events = await processor.process_log_entry(log_entry)
        print(f"Processed {len(events)} security events")
    
    # Generate security report
    report = processor.generate_security_report(hours=1)
    print(f"Security report: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Best Practices Summary

### 1. Credential Management
- **Never hard-code credentials**: Use environment variables or secure key management
- **Encrypt stored credentials**: Use strong encryption for credential storage
- **Rotate credentials regularly**: Implement automatic credential rotation
- **Use least privilege principle**: Grant minimum necessary permissions

### 2. Data Protection
- **Encrypt sensitive data**: Encrypt quantum algorithms and training data
- **Sanitize circuit parameters**: Add controlled noise to prevent reverse engineering
- **Secure data transmission**: Use TLS/SSL for all data in transit
- **Implement data classification**: Classify and handle data according to sensitivity

### 3. Access Control
- **Implement RBAC**: Use role-based access control for quantum resources
- **Monitor access patterns**: Detect unusual access patterns and suspicious activities
- **Session management**: Implement secure session handling with timeouts
- **Audit access logs**: Maintain comprehensive audit trails

### 4. Cost and Resource Management
- **Set budget limits**: Implement cost controls and budget alerts
- **Monitor resource usage**: Track quantum resource consumption
- **Implement quotas**: Set usage quotas per user/team
- **Cost allocation**: Track costs by project and user

### 5. Security Monitoring
- **Continuous monitoring**: Monitor quantum resources 24/7
- **Automated alerting**: Set up alerts for security events
- **Incident response**: Have procedures for security incident response
- **Regular security assessments**: Conduct periodic security reviews

## 📚 Additional Resources

### Quantum Cloud Provider Documentation
- [IBM Quantum Network Security](https://quantum-computing.ibm.com/docs/manage/account/security)
- [AWS Braket Security Best Practices](https://docs.aws.amazon.com/braket/latest/developerguide/security.html)
- [IonQ API Security Guide](https://docs.ionq.com/#section/Authentication)
- [Rigetti Cloud Security](https://docs.rigetti.com/qcs/guides/authentication)

### Security Frameworks and Standards
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO/IEC 27001](https://www.iso.org/isoiec-27001-information-security.html)
- [OWASP Cloud Security](https://owasp.org/www-project-cloud-security/)
- [CSA Cloud Controls Matrix](https://cloudsecurityalliance.org/research/cloud-controls-matrix/)

### Quantum Security Research
- [Quantum Cryptography and Security](https://arxiv.org/abs/quant-ph/0106096)
- [Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [Quantum Key Distribution Security](https://doi.org/10.1038/npjqi.2016.25)
- [Quantum Computing Security Implications](https://arxiv.org/abs/2009.03788)