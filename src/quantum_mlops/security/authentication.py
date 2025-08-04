"""Authentication and JWT management for quantum MLOps workbench."""

import os
import jwt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import logging

try:
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User account representation."""
    
    username: str
    email: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_admin: bool = False
    roles: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = ["user"]
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class AuthToken:
    """Authentication token representation."""
    
    token: str
    user_id: str
    expires_at: datetime
    token_type: str = "access"  # access, refresh, api_key
    scopes: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at


class PasswordManager:
    """Secure password hashing and verification."""
    
    def __init__(self):
        """Initialize password manager."""
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            logger.warning("passlib not available, using fallback hashing")
            
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if PASSLIB_AVAILABLE:
            return self.pwd_context.hash(password)
        else:
            # Fallback - not as secure
            salt = secrets.token_hex(16)
            return f"fallback:{salt}:{hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()}"
            
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if PASSLIB_AVAILABLE:
            return self.pwd_context.verify(password, hashed)
        else:
            if not hashed.startswith("fallback:"):
                return False
            _, salt, expected = hashed.split(":", 2)
            actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            return secrets.compare_digest(actual, expected)


class JWTManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256"):
        """Initialize JWT manager."""
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY')
        if not self.secret_key:
            raise ValueError("JWT secret key required")
        self.algorithm = algorithm
        
    def create_access_token(self, user_id: str, expires_delta: timedelta = None, 
                          scopes: List[str] = None, **kwargs) -> str:
        """Create JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(hours=1)
            
        expires_at = datetime.utcnow() + expires_delta
        
        payload = {
            "sub": user_id,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "type": "access",
            "scopes": scopes or [],
            **kwargs
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def create_refresh_token(self, user_id: str, expires_delta: timedelta = None) -> str:
        """Create JWT refresh token."""
        if expires_delta is None:
            expires_delta = timedelta(days=30)
            
        expires_at = datetime.utcnow() + expires_delta
        
        payload = {
            "sub": user_id,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
            
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.decode_token(refresh_token)
        
        if payload.get('type') != 'refresh':
            raise ValueError("Invalid refresh token")
            
        user_id = payload.get('sub')
        if not user_id:
            raise ValueError("Invalid token payload")
            
        return self.create_access_token(user_id)


class APIKeyManager:
    """API key management for service authentication."""
    
    def __init__(self, secret_key: str = None):
        """Initialize API key manager."""
        self.secret_key = secret_key or os.getenv('API_KEY_SALT', 'quantum-mlops')
        
    def generate_api_key(self, user_id: str, name: str = None, 
                        scopes: List[str] = None) -> Dict[str, str]:
        """Generate API key for user."""
        # Generate random key
        key_bytes = secrets.token_bytes(32)
        api_key = secrets.token_urlsafe(32)
        
        # Create verifiable hash
        key_data = f"{user_id}:{name or 'default'}:{datetime.utcnow().isoformat()}"
        signature = hashlib.hmac.new(
            self.secret_key.encode(), 
            f"{api_key}:{key_data}".encode(), 
            hashlib.sha256
        ).hexdigest()
        
        return {
            "api_key": f"qml_{api_key}",
            "signature": signature,
            "user_id": user_id,
            "name": name or "default",
            "scopes": scopes or ["read"],
            "created_at": datetime.utcnow().isoformat()
        }
        
    def verify_api_key(self, api_key: str, stored_signature: str, 
                      user_id: str, name: str, created_at: str) -> bool:
        """Verify API key authenticity."""
        if not api_key.startswith("qml_"):
            return False
            
        clean_key = api_key[4:]  # Remove qml_ prefix
        key_data = f"{user_id}:{name}:{created_at}"
        expected_signature = hashlib.hmac.new(
            self.secret_key.encode(),
            f"{clean_key}:{key_data}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return secrets.compare_digest(expected_signature, stored_signature)


class AuthenticationManager:
    """Main authentication manager."""
    
    def __init__(self, secret_key: str = None):
        """Initialize authentication manager."""
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager(secret_key)
        self.api_key_manager = APIKeyManager(secret_key)
        self._users: Dict[str, User] = {}
        self._active_tokens: Dict[str, AuthToken] = {}
        
    def create_user(self, username: str, email: str, password: str, 
                   is_admin: bool = False, roles: List[str] = None) -> User:
        """Create a new user account."""
        if username in self._users:
            raise ValueError(f"User {username} already exists")
            
        hashed_password = self.password_manager.hash_password(password)
        
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.utcnow(),
            is_admin=is_admin,
            roles=roles or ["user"]
        )
        
        self._users[username] = user
        logger.info(f"Created user: {username}")
        return user
        
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        user = self._users.get(username)
        if not user or not user.is_active:
            return None
            
        if not self.password_manager.verify_password(password, user.hashed_password):
            return None
            
        # Update last login
        user.last_login = datetime.utcnow()
        logger.info(f"User authenticated: {username}")
        return user
        
    def login(self, username: str, password: str) -> Dict[str, str]:
        """Login user and return tokens."""
        user = self.authenticate_user(username, password)
        if not user:
            raise ValueError("Invalid credentials")
            
        # Generate tokens
        access_token = self.jwt_manager.create_access_token(
            user_id=username,
            scopes=user.roles
        )
        refresh_token = self.jwt_manager.create_refresh_token(username)
        
        # Store active tokens
        access_token_obj = AuthToken(
            token=access_token,
            user_id=username,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            token_type="access",
            scopes=user.roles
        )
        
        refresh_token_obj = AuthToken(
            token=refresh_token,
            user_id=username,
            expires_at=datetime.utcnow() + timedelta(days=30),
            token_type="refresh"
        )
        
        self._active_tokens[access_token] = access_token_obj
        self._active_tokens[refresh_token] = refresh_token_obj
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 3600
        }
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode authentication token."""
        try:
            payload = self.jwt_manager.decode_token(token)
            
            # Check if token is in active tokens
            token_obj = self._active_tokens.get(token)
            if token_obj and token_obj.is_expired():
                # Remove expired token
                del self._active_tokens[token]
                return None
                
            return payload
            
        except ValueError:
            return None
            
    def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token."""
        if token in self._active_tokens:
            del self._active_tokens[token]
            return True
        return False
        
    def revoke_all_user_tokens(self, username: str) -> int:
        """Revoke all tokens for a user."""
        count = 0
        tokens_to_remove = []
        
        for token, token_obj in self._active_tokens.items():
            if token_obj.user_id == username:
                tokens_to_remove.append(token)
                count += 1
                
        for token in tokens_to_remove:
            del self._active_tokens[token]
            
        return count
        
    def create_api_key(self, username: str, name: str = None, 
                      scopes: List[str] = None) -> Dict[str, str]:
        """Create API key for user."""
        user = self._users.get(username)
        if not user:
            raise ValueError(f"User {username} not found")
            
        return self.api_key_manager.generate_api_key(username, name, scopes)
        
    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens."""
        expired_tokens = []
        
        for token, token_obj in self._active_tokens.items():
            if token_obj.is_expired():
                expired_tokens.append(token)
                
        for token in expired_tokens:
            del self._active_tokens[token]
            
        return len(expired_tokens)
        
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (without sensitive data)."""
        user = self._users.get(username)
        if not user:
            return None
            
        return {
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "roles": user.roles,
            "metadata": user.metadata
        }


class JWTAuthenticator:
    """JWT-based request authenticator."""
    
    def __init__(self, auth_manager: AuthenticationManager):
        """Initialize JWT authenticator."""
        self.auth_manager = auth_manager
        
    def require_auth(self, scopes: List[str] = None):
        """Decorator to require authentication."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract token from request headers
                # This would typically be integrated with your web framework
                token = kwargs.get('auth_token') or kwargs.get('token')
                if not token:
                    raise ValueError("Authentication token required")
                    
                payload = self.auth_manager.verify_token(token)
                if not payload:
                    raise ValueError("Invalid or expired token")
                    
                # Check scopes if required
                if scopes:
                    token_scopes = payload.get('scopes', [])
                    if not any(scope in token_scopes for scope in scopes):
                        raise ValueError("Insufficient permissions")
                        
                # Add user info to kwargs
                kwargs['current_user'] = payload.get('sub')
                kwargs['user_scopes'] = payload.get('scopes', [])
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def require_admin(self):
        """Decorator to require admin access."""
        return self.require_auth(scopes=['admin'])
        
    def require_quantum_access(self):
        """Decorator to require quantum backend access."""
        return self.require_auth(scopes=['quantum', 'admin'])


# Global authentication manager
_global_auth_manager: Optional[AuthenticationManager] = None

def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager."""
    global _global_auth_manager
    if _global_auth_manager is None:
        _global_auth_manager = AuthenticationManager()
    return _global_auth_manager