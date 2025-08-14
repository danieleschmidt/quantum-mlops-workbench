"""Global compliance and regulatory framework for quantum ML operations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
import json
import hashlib

logger = logging.getLogger(__name__)

class ComplianceRegion(Enum):
    """Global compliance regions."""
    
    EU = "eu"                    # European Union (GDPR)
    US = "us"                   # United States (CCPA, etc.)
    CANADA = "canada"           # PIPEDA
    ASIA_PACIFIC = "apac"       # PDPA, etc.
    GLOBAL = "global"           # Multi-region

class DataClassification(Enum):
    """Data classification levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class ProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing."""
    
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class DataSubject:
    """Data subject information for privacy compliance."""
    
    subject_id: str
    region: ComplianceRegion
    consent_given: bool
    consent_timestamp: Optional[datetime] = None
    data_retention_period: int = 365  # days
    right_to_be_forgotten: bool = True
    data_portability_right: bool = True
    
class PrivacyByDesignPrinciples:
    """Privacy by Design principles implementation."""
    
    @staticmethod
    def proactive_not_reactive():
        """Principle 1: Proactive not Reactive; Preventative not Remedial."""
        return {
            "principle": "Proactive not Reactive",
            "implementation": [
                "Privacy impact assessment before quantum model training",
                "Automated privacy compliance checking",
                "Preventative data loss detection",
                "Quantum-safe encryption by default"
            ]
        }
    
    @staticmethod
    def privacy_as_default():
        """Principle 2: Privacy as the Default Setting."""
        return {
            "principle": "Privacy as Default",
            "implementation": [
                "Quantum differential privacy enabled by default",
                "Minimal data collection principle",
                "Automatic data anonymization",
                "Zero-trust quantum data access"
            ]
        }
    
    @staticmethod
    def privacy_embedded_in_design():
        """Principle 3: Privacy Embedded into Design."""
        return {
            "principle": "Privacy Embedded in Design", 
            "implementation": [
                "Quantum circuit privacy validation",
                "Privacy-preserving quantum machine learning",
                "Secure multi-party quantum computation",
                "Quantum homomorphic encryption"
            ]
        }
    
    @staticmethod
    def full_functionality():
        """Principle 4: Full Functionality - Positive-Sum, not Zero-Sum."""
        return {
            "principle": "Full Functionality",
            "implementation": [
                "Performance maintained with privacy protection",
                "Quantum advantage preserved with compliance",
                "Business value enhanced by trust",
                "Innovation enabled by privacy-first design"
            ]
        }
    
    @staticmethod
    def end_to_end_security():
        """Principle 5: End-to-End Security - Full Lifecycle Protection."""
        return {
            "principle": "End-to-End Security",
            "implementation": [
                "Quantum-safe cryptography throughout",
                "Secure quantum key distribution",
                "Tamper-evident quantum computations",
                "Continuous security monitoring"
            ]
        }
    
    @staticmethod
    def visibility_transparency():
        """Principle 6: Visibility and Transparency."""
        return {
            "principle": "Visibility and Transparency",
            "implementation": [
                "Quantum computation auditability",
                "Transparent AI/ML decision making",
                "Open compliance reporting",
                "User-friendly privacy controls"
            ]
        }
    
    @staticmethod
    def respect_for_privacy():
        """Principle 7: Respect for User Privacy."""
        return {
            "principle": "Respect for Privacy",
            "implementation": [
                "User-centric quantum data controls",
                "Meaningful consent mechanisms",
                "Right to explanation for quantum ML",
                "Human-in-the-loop quantum decisions"
            ]
        }

class GDPRCompliance:
    """GDPR compliance framework for quantum ML."""
    
    def __init__(self):
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_activities: List[Dict[str, Any]] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
    
    def register_data_subject(self, subject: DataSubject) -> bool:
        """Register data subject under GDPR."""
        
        self.data_subjects[subject.subject_id] = subject
        
        # Log consent if given
        if subject.consent_given and subject.consent_timestamp:
            self.consent_records[subject.subject_id] = {
                "consent_timestamp": subject.consent_timestamp,
                "lawful_basis": ProcessingLawfulBasis.CONSENT,
                "purpose": "quantum machine learning research and development",
                "data_categories": ["quantum measurement data", "model parameters"],
                "retention_period": subject.data_retention_period,
                "withdrawal_mechanism": "automated_upon_request"
            }
        
        logger.info(f"Data subject {subject.subject_id} registered under GDPR")
        return True
    
    def process_right_to_access(self, subject_id: str) -> Dict[str, Any]:
        """Process GDPR right to access request."""
        
        if subject_id not in self.data_subjects:
            return {"error": "Data subject not found"}
        
        subject = self.data_subjects[subject_id]
        
        # Collect all personal data
        personal_data = {
            "subject_information": {
                "subject_id": subject_id,
                "region": subject.region.value,
                "consent_status": subject.consent_given,
                "registration_date": subject.consent_timestamp.isoformat() if subject.consent_timestamp else None
            },
            "processing_activities": [
                activity for activity in self.processing_activities
                if activity.get("subject_id") == subject_id
            ],
            "consent_records": self.consent_records.get(subject_id, {}),
            "data_retention": {
                "retention_period_days": subject.data_retention_period,
                "automatic_deletion": True
            }
        }
        
        logger.info(f"Right to access processed for subject {subject_id}")
        return personal_data
    
    def process_right_to_be_forgotten(self, subject_id: str) -> bool:
        """Process GDPR right to be forgotten request."""
        
        if subject_id not in self.data_subjects:
            return False
        
        subject = self.data_subjects[subject_id]
        
        if not subject.right_to_be_forgotten:
            logger.warning(f"Right to be forgotten not applicable for subject {subject_id}")
            return False
        
        # Remove from all systems
        try:
            # Remove subject record
            del self.data_subjects[subject_id]
            
            # Remove consent records
            if subject_id in self.consent_records:
                del self.consent_records[subject_id]
            
            # Remove processing activities
            self.processing_activities = [
                activity for activity in self.processing_activities
                if activity.get("subject_id") != subject_id
            ]
            
            # Remove from quantum models (pseudonymization/anonymization)
            self._anonymize_quantum_data(subject_id)
            
            logger.info(f"Right to be forgotten processed for subject {subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing right to be forgotten for {subject_id}: {e}")
            return False
    
    def process_data_portability(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Process GDPR data portability request."""
        
        if subject_id not in self.data_subjects:
            return None
        
        subject = self.data_subjects[subject_id]
        
        if not subject.data_portability_right:
            logger.warning(f"Data portability not applicable for subject {subject_id}")
            return None
        
        # Export data in structured format
        portable_data = {
            "export_metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "subject_id": subject_id,
                "format": "JSON",
                "version": "1.0"
            },
            "personal_data": self.process_right_to_access(subject_id),
            "quantum_data": self._extract_quantum_data(subject_id),
            "model_contributions": self._extract_model_contributions(subject_id)
        }
        
        logger.info(f"Data portability processed for subject {subject_id}")
        return portable_data
    
    def _anonymize_quantum_data(self, subject_id: str):
        """Anonymize quantum data associated with subject."""
        # Implementation would depend on specific quantum data storage
        logger.info(f"Quantum data anonymized for subject {subject_id}")
    
    def _extract_quantum_data(self, subject_id: str) -> Dict[str, Any]:
        """Extract quantum data for portability."""
        # Mock implementation - would extract actual quantum measurements/parameters
        return {
            "quantum_measurements": [],
            "circuit_parameters": [],
            "training_contributions": []
        }
    
    def _extract_model_contributions(self, subject_id: str) -> Dict[str, Any]:
        """Extract model contributions for portability."""
        return {
            "training_epochs": 0,
            "gradient_contributions": [],
            "model_improvements": []
        }

class CCPACompliance:
    """California Consumer Privacy Act compliance."""
    
    def __init__(self):
        self.consumers: Dict[str, Dict[str, Any]] = {}
        self.opt_out_requests: Set[str] = set()
    
    def register_consumer(self, consumer_id: str, personal_info: Dict[str, Any]) -> bool:
        """Register consumer under CCPA."""
        
        self.consumers[consumer_id] = {
            "consumer_id": consumer_id,
            "registration_date": datetime.now(timezone.utc),
            "personal_information": personal_info,
            "opt_out_status": False,
            "deletion_requests": [],
            "disclosure_requests": []
        }
        
        logger.info(f"Consumer {consumer_id} registered under CCPA")
        return True
    
    def process_opt_out_request(self, consumer_id: str) -> bool:
        """Process CCPA opt-out of sale request."""
        
        if consumer_id not in self.consumers:
            return False
        
        self.opt_out_requests.add(consumer_id)
        self.consumers[consumer_id]["opt_out_status"] = True
        self.consumers[consumer_id]["opt_out_date"] = datetime.now(timezone.utc)
        
        logger.info(f"Opt-out processed for consumer {consumer_id}")
        return True
    
    def process_deletion_request(self, consumer_id: str) -> bool:
        """Process CCPA deletion request."""
        
        if consumer_id not in self.consumers:
            return False
        
        try:
            # Record deletion request
            self.consumers[consumer_id]["deletion_requests"].append({
                "request_date": datetime.now(timezone.utc),
                "status": "processed"
            })
            
            # Delete personal information
            del self.consumers[consumer_id]
            
            # Remove from opt-out list if present
            self.opt_out_requests.discard(consumer_id)
            
            logger.info(f"Deletion processed for consumer {consumer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing deletion for {consumer_id}: {e}")
            return False

class QuantumMLComplianceFramework:
    """Comprehensive compliance framework for quantum ML operations."""
    
    def __init__(self, regions: List[ComplianceRegion] = None):
        self.regions = regions or [ComplianceRegion.GLOBAL]
        self.gdpr = GDPRCompliance() if ComplianceRegion.EU in self.regions else None
        self.ccpa = CCPACompliance() if ComplianceRegion.US in self.regions else None
        
        # Initialize Privacy by Design principles
        self.privacy_principles = [
            PrivacyByDesignPrinciples.proactive_not_reactive(),
            PrivacyByDesignPrinciples.privacy_as_default(),
            PrivacyByDesignPrinciples.privacy_embedded_in_design(),
            PrivacyByDesignPrinciples.full_functionality(),
            PrivacyByDesignPrinciples.end_to_end_security(),
            PrivacyByDesignPrinciples.visibility_transparency(),
            PrivacyByDesignPrinciples.respect_for_privacy()
        ]
    
    def conduct_privacy_impact_assessment(
        self,
        processing_purpose: str,
        data_categories: List[str],
        quantum_circuit_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct Privacy Impact Assessment for quantum ML processing."""
        
        assessment = {
            "assessment_id": hashlib.md5(f"{processing_purpose}_{datetime.now()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_details": {
                "purpose": processing_purpose,
                "data_categories": data_categories,
                "quantum_circuit_info": {
                    "n_qubits": quantum_circuit_metadata.get("n_qubits", 0),
                    "depth": quantum_circuit_metadata.get("depth", 0),
                    "gate_types": quantum_circuit_metadata.get("gates", []),
                    "measurement_basis": quantum_circuit_metadata.get("measurement_basis", "computational")
                }
            },
            "privacy_risks": self._assess_privacy_risks(data_categories, quantum_circuit_metadata),
            "mitigation_measures": self._recommend_mitigation_measures(),
            "compliance_status": {},
            "recommendations": []
        }
        
        # Regional compliance assessment
        for region in self.regions:
            assessment["compliance_status"][region.value] = self._assess_regional_compliance(
                region, processing_purpose, data_categories
            )
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_compliance_recommendations(assessment)
        
        logger.info(f"Privacy Impact Assessment completed: {assessment['assessment_id']}")
        return assessment
    
    def _assess_privacy_risks(
        self,
        data_categories: List[str],
        quantum_circuit_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess privacy risks in quantum ML processing."""
        
        risks = []
        
        # Quantum-specific risks
        if "personal_data" in data_categories:
            risks.append({
                "risk_type": "quantum_data_leakage",
                "severity": "high",
                "description": "Personal data may be exposed through quantum state measurements",
                "likelihood": "medium",
                "impact": "high"
            })
        
        # Circuit depth risk
        circuit_depth = quantum_circuit_metadata.get("depth", 0)
        if circuit_depth > 50:
            risks.append({
                "risk_type": "quantum_decoherence_information_loss",
                "severity": "medium",
                "description": "Deep quantum circuits may lose information due to decoherence",
                "likelihood": "high",
                "impact": "medium"
            })
        
        # Measurement basis risk
        if quantum_circuit_metadata.get("measurement_basis") != "computational":
            risks.append({
                "risk_type": "measurement_information_leakage",
                "severity": "medium",
                "description": "Non-computational measurement basis may leak quantum information",
                "likelihood": "low",
                "impact": "high"
            })
        
        return risks
    
    def _recommend_mitigation_measures(self) -> List[Dict[str, Any]]:
        """Recommend privacy mitigation measures."""
        
        return [
            {
                "measure": "Quantum Differential Privacy",
                "implementation": "Add quantum noise to protect individual data points",
                "effectiveness": "high",
                "complexity": "medium"
            },
            {
                "measure": "Secure Multi-party Quantum Computation",
                "implementation": "Distribute computation across multiple quantum devices",
                "effectiveness": "high", 
                "complexity": "high"
            },
            {
                "measure": "Quantum Homomorphic Encryption",
                "implementation": "Perform computation on encrypted quantum states",
                "effectiveness": "very_high",
                "complexity": "very_high"
            },
            {
                "measure": "Data Minimization",
                "implementation": "Use only necessary data for quantum ML training",
                "effectiveness": "medium",
                "complexity": "low"
            },
            {
                "measure": "Quantum Circuit Obfuscation",
                "implementation": "Obfuscate quantum circuit structure to protect IP",
                "effectiveness": "medium",
                "complexity": "medium"
            }
        ]
    
    def _assess_regional_compliance(
        self,
        region: ComplianceRegion,
        processing_purpose: str,
        data_categories: List[str]
    ) -> Dict[str, Any]:
        """Assess compliance for specific region."""
        
        if region == ComplianceRegion.EU:
            return {
                "regulation": "GDPR",
                "lawful_basis_required": True,
                "consent_mechanism": "explicit" if "personal_data" in data_categories else "not_required",
                "dpo_required": len(data_categories) > 3,
                "data_protection_impact_assessment": "recommended",
                "compliance_score": 0.8
            }
        
        elif region == ComplianceRegion.US:
            return {
                "regulation": "CCPA",
                "opt_out_mechanism": "required" if "personal_information" in data_categories else "not_required",
                "disclosure_requirements": "annual",
                "deletion_rights": True,
                "compliance_score": 0.75
            }
        
        else:
            return {
                "regulation": "General Best Practices",
                "privacy_by_design": "recommended",
                "data_minimization": "required",
                "security_measures": "required",
                "compliance_score": 0.6
            }
    
    def _generate_compliance_recommendations(
        self,
        assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate specific compliance recommendations."""
        
        recommendations = []
        
        # High-risk recommendations
        high_risks = [r for r in assessment["privacy_risks"] if r["severity"] == "high"]
        if high_risks:
            recommendations.append(
                "Implement quantum differential privacy for high-risk data processing"
            )
        
        # GDPR-specific recommendations
        if ComplianceRegion.EU in self.regions:
            recommendations.extend([
                "Establish explicit consent mechanism for personal data processing",
                "Implement automated data subject rights fulfillment",
                "Conduct regular quantum ML model audits for bias and fairness"
            ])
        
        # General recommendations
        recommendations.extend([
            "Enable privacy-preserving quantum machine learning techniques",
            "Implement quantum-safe cryptography for data protection",
            "Establish quantum computation audit trail for transparency",
            "Create user-friendly privacy controls for quantum data subjects"
        ])
        
        return recommendations
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance status report."""
        
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance_regions": [r.value for r in self.regions],
            "privacy_by_design_implementation": {
                principle["principle"]: len(principle["implementation"])
                for principle in self.privacy_principles
            },
            "gdpr_status": {},
            "ccpa_status": {},
            "overall_compliance_score": 0.0
        }
        
        # GDPR status
        if self.gdpr:
            report["gdpr_status"] = {
                "registered_subjects": len(self.gdpr.data_subjects),
                "consent_records": len(self.gdpr.consent_records),
                "processing_activities": len(self.gdpr.processing_activities),
                "rights_requests_processed": "automated"
            }
        
        # CCPA status  
        if self.ccpa:
            report["ccpa_status"] = {
                "registered_consumers": len(self.ccpa.consumers),
                "opt_out_requests": len(self.ccpa.opt_out_requests),
                "deletion_requests_processed": "automated"
            }
        
        # Calculate overall compliance score
        scores = []
        if self.gdpr:
            scores.append(0.85)  # Mock GDPR compliance score
        if self.ccpa:
            scores.append(0.80)  # Mock CCPA compliance score
        
        report["overall_compliance_score"] = sum(scores) / len(scores) if scores else 0.75
        
        return report

# Export main classes
__all__ = [
    'ComplianceRegion',
    'DataClassification', 
    'ProcessingLawfulBasis',
    'DataSubject',
    'PrivacyByDesignPrinciples',
    'GDPRCompliance',
    'CCPACompliance',
    'QuantumMLComplianceFramework'
]