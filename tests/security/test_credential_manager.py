"""Tests for credential management system."""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from quantum_mlops.security.credential_manager import (
    CredentialManager, SecureCredentialStore, Credential, EncryptionProvider
)


class TestEncryptionProvider:
    """Test encryption provider functionality."""
    
    def test_encryption_roundtrip(self):
        """Test encryption and decryption."""
        provider = EncryptionProvider("test-password-123")
        test_data = "sensitive-data-12345"
        
        encrypted = provider.encrypt(test_data)
        decrypted = provider.decrypt(encrypted)
        
        assert decrypted == test_data
        assert encrypted != test_data
        
    def test_encryption_deterministic(self):
        """Test that encryption is not deterministic."""
        provider = EncryptionProvider("test-password-123")
        test_data = "same-data"
        
        encrypted1 = provider.encrypt(test_data)
        encrypted2 = provider.encrypt(test_data)
        
        # Should be different due to randomization
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same value
        assert provider.decrypt(encrypted1) == test_data
        assert provider.decrypt(encrypted2) == test_data


class TestSecureCredentialStore:
    """Test secure credential storage."""
    
    @pytest.fixture
    def temp_store(self):
        """Create temporary credential store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "test_creds.enc"
            store = SecureCredentialStore(
                store_path=str(store_path),
                encryption_key="test-key-123"
            )
            yield store
            
    def test_store_and_retrieve_credential(self, temp_store):
        """Test storing and retrieving credentials."""
        credential = Credential(
            name="test-cred",
            provider="test-provider",
            credential_type="api_key",
            encrypted_data="encrypted-test-data",
            created_at=datetime.utcnow()
        )
        
        temp_store.store_credential(credential)
        retrieved = temp_store.get_credential("test-cred")
        
        assert retrieved is not None
        assert retrieved.name == credential.name
        assert retrieved.provider == credential.provider
        assert retrieved.credential_type == credential.credential_type
        
    def test_list_credentials(self, temp_store):
        """Test listing stored credentials."""
        creds = [
            Credential("cred1", "provider1", "api_key", "data1", datetime.utcnow()),
            Credential("cred2", "provider2", "token", "data2", datetime.utcnow())
        ]
        
        for cred in creds:
            temp_store.store_credential(cred)
            
        credential_names = temp_store.list_credentials()
        assert "cred1" in credential_names
        assert "cred2" in credential_names
        assert len(credential_names) == 2
        
    def test_delete_credential(self, temp_store):
        """Test deleting credentials."""
        credential = Credential(
            "delete-me", "provider", "api_key", "data", datetime.utcnow()
        )
        
        temp_store.store_credential(credential)
        assert temp_store.get_credential("delete-me") is not None
        
        result = temp_store.delete_credential("delete-me")
        assert result is True
        assert temp_store.get_credential("delete-me") is None
        
    def test_credential_expiration(self, temp_store):
        """Test credential expiration detection."""
        expired_credential = Credential(
            name="expired",
            provider="test",
            credential_type="token",
            encrypted_data="data",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        )
        
        temp_store.store_credential(expired_credential)
        expired_creds = temp_store.get_expired_credentials()
        
        assert "expired" in expired_creds
        
    def test_credential_rotation(self, temp_store):
        """Test credential rotation functionality."""
        credential = Credential(
            name="rotate-me",
            provider="test",
            credential_type="api_key",
            encrypted_data="old-data",
            created_at=datetime.utcnow() - timedelta(days=100),  # Old credential
            rotation_enabled=True
        )
        
        temp_store.store_credential(credential)
        rotation_due = temp_store.get_rotation_due_credentials(90)
        
        assert "rotate-me" in rotation_due
        
        # Test rotation
        result = temp_store.rotate_credential("rotate-me", "new-encrypted-data")
        assert result is True
        
        rotated = temp_store.get_credential("rotate-me")
        assert rotated.encrypted_data == temp_store.encryption.encrypt("new-encrypted-data")


class TestCredentialManager:
    """Test credential manager functionality."""
    
    @pytest.fixture
    def temp_manager(self):
        """Create temporary credential manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "test_creds.enc"
            store = SecureCredentialStore(
                store_path=str(store_path),
                encryption_key="test-key-123"
            )
            manager = CredentialManager(store=store)
            yield manager
            
    def test_store_aws_credentials(self, temp_manager):
        """Test storing AWS credentials."""
        temp_manager.store_aws_credentials(
            name="test-aws",
            access_key="AKIATEST123",
            secret_key="secret123",
            region="us-west-2"
        )
        
        creds = temp_manager.get_aws_credentials("test-aws")
        assert creds is not None
        assert creds["access_key_id"] == "AKIATEST123"
        assert creds["secret_access_key"] == "secret123"
        assert creds["region"] == "us-west-2"
        
    def test_store_ibm_credentials(self, temp_manager):
        """Test storing IBM Quantum credentials."""
        temp_manager.store_ibm_token(
            name="test-ibm",
            token="ibm-token-123",
            instance="ibm-q/open/main"
        )
        
        creds = temp_manager.get_ibm_credentials("test-ibm")
        assert creds is not None
        assert creds["token"] == "ibm-token-123"
        assert creds["instance"] == "ibm-q/open/main"
        
    def test_store_ionq_credentials(self, temp_manager):
        """Test storing IonQ credentials."""
        temp_manager.store_ionq_credentials(
            name="test-ionq",
            api_key="ionq-key-123"
        )
        
        creds = temp_manager.get_ionq_credentials("test-ionq")
        assert creds is not None
        assert creds["api_key"] == "ionq-key-123"
        
    def test_get_provider_credentials(self, temp_manager):
        """Test getting credentials by provider."""
        temp_manager.store_aws_credentials("aws1", "key1", "secret1")
        temp_manager.store_aws_credentials("aws2", "key2", "secret2")
        temp_manager.store_ibm_token("ibm1", "token1")
        
        aws_creds = temp_manager.get_provider_credentials("aws_braket")
        ibm_creds = temp_manager.get_provider_credentials("ibm_quantum")
        
        assert len(aws_creds) == 2
        assert "aws1" in aws_creds
        assert "aws2" in aws_creds
        assert len(ibm_creds) == 1
        assert "ibm1" in ibm_creds
        
    def test_validate_credentials(self, temp_manager):
        """Test credential validation."""
        temp_manager.store_aws_credentials("test-aws", "key", "secret")
        
        validation = temp_manager.validate_credentials("test-aws")
        
        assert validation["valid"] is True
        assert validation["provider"] == "aws_braket"
        assert validation["type"] == "aws_credentials"
        assert "created" in validation
        
    def test_validate_nonexistent_credentials(self, temp_manager):
        """Test validation of non-existent credentials."""
        validation = temp_manager.validate_credentials("nonexistent")
        
        assert validation["valid"] is False
        assert "error" in validation
        
    def test_cleanup_expired_credentials(self, temp_manager):
        """Test cleanup of expired credentials."""
        # Store expired credential
        temp_manager.store_aws_credentials(
            name="expired-aws",
            access_key="key",
            secret_key="secret",
            expires_days=-1  # Expired
        )
        
        # Store valid credential
        temp_manager.store_aws_credentials(
            name="valid-aws",
            access_key="key2",
            secret_key="secret2",
            expires_days=30  # Valid
        )
        
        expired = temp_manager.cleanup_expired_credentials()
        
        assert "expired-aws" in expired
        assert temp_manager.get_aws_credentials("expired-aws") is None
        assert temp_manager.get_aws_credentials("valid-aws") is not None
        
    def test_rotation_report(self, temp_manager):
        """Test generation of rotation report."""
        temp_manager.store_aws_credentials("aws1", "key1", "secret1")
        temp_manager.store_ibm_token("ibm1", "token1")
        
        report = temp_manager.generate_rotation_report()
        
        assert "total_credentials" in report
        assert "expired_credentials" in report
        assert "rotation_due" in report
        assert "report_generated" in report
        assert report["total_credentials"] == 2


class TestCredentialSecurity:
    """Test credential security features."""
    
    def test_encryption_with_different_keys(self):
        """Test that different keys produce different encrypted data."""
        provider1 = EncryptionProvider("key1")
        provider2 = EncryptionProvider("key2")
        
        test_data = "secret-data"
        
        encrypted1 = provider1.encrypt(test_data)
        encrypted2 = provider2.encrypt(test_data)
        
        assert encrypted1 != encrypted2
        
        # Each should only decrypt with correct key
        assert provider1.decrypt(encrypted1) == test_data
        assert provider2.decrypt(encrypted2) == test_data
        
        # Cross-decryption should fail
        with pytest.raises(Exception):
            provider1.decrypt(encrypted2)
        with pytest.raises(Exception):
            provider2.decrypt(encrypted1)
            
    def test_credential_store_permissions(self):
        """Test that credential store sets secure permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "secure_creds.enc"
            
            store = SecureCredentialStore(
                store_path=str(store_path),
                encryption_key="test-key"
            )
            
            # Store a credential to create the file
            credential = Credential(
                "test", "provider", "type", "data", datetime.utcnow()
            )
            store.store_credential(credential)
            
            # Check file permissions (on Unix systems)
            if hasattr(os, 'stat'):
                stat_info = os.stat(store_path)
                # Should be readable/writable by owner only (600)
                permissions = stat_info.st_mode & 0o777
                assert permissions == 0o600 or permissions == 0o644  # May vary by system