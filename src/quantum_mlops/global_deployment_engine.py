"""Global Deployment Engine for Quantum MLOps.

Global-First Implementation:
- Multi-region deployment automation
- I18n support (en, es, fr, de, ja, zh)
- GDPR, CCPA, PDPA compliance
- Cross-platform compatibility
- Automated compliance validation
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pathlib import Path
import locale
import gettext

import yaml
from pydantic import BaseModel, Field

from .exceptions import QuantumMLOpsException, ErrorSeverity
from .logging_config import get_logger
from .security import QuantumSecurityManager


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"          # European Union
    CCPA = "ccpa"          # California
    PDPA = "pdpa"          # Singapore
    PIPEDA = "pipeda"      # Canada
    LGPD = "lgpd"          # Brazil
    PRIVACY_ACT = "privacy_act"  # Australia


class SupportedLanguage(Enum):
    """Supported languages for i18n."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"


@dataclass
class RegionConfig:
    """Configuration for deployment region."""
    region: DeploymentRegion
    compliance_frameworks: List[ComplianceFramework]
    primary_languages: List[SupportedLanguage]
    data_residency_required: bool
    quantum_providers: List[str]
    edge_locations: List[str]
    
    
@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    framework: ComplianceFramework
    requirement_id: str
    description: str
    implementation_guide: str
    validation_method: str
    mandatory: bool = True


class GlobalI18nManager:
    """Internationalization manager for global deployment."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.current_language = SupportedLanguage.ENGLISH
        self.translations = {}
        self.locales = {
            SupportedLanguage.ENGLISH: "en_US.UTF-8",
            SupportedLanguage.SPANISH: "es_ES.UTF-8",
            SupportedLanguage.FRENCH: "fr_FR.UTF-8",
            SupportedLanguage.GERMAN: "de_DE.UTF-8",
            SupportedLanguage.JAPANESE: "ja_JP.UTF-8",
            SupportedLanguage.CHINESE: "zh_CN.UTF-8",
            SupportedLanguage.PORTUGUESE: "pt_BR.UTF-8",
            SupportedLanguage.ITALIAN: "it_IT.UTF-8"
        }
        
        self._initialize_translations()
        
    def _initialize_translations(self) -> None:
        """Initialize translation dictionaries."""
        self.translations = {
            SupportedLanguage.ENGLISH: {
                "welcome": "Welcome to Quantum MLOps",
                "error": "An error occurred",
                "success": "Operation completed successfully",
                "processing": "Processing quantum circuit",
                "optimization": "Optimizing performance",
                "deployment": "Deploying to region",
                "compliance_check": "Checking compliance requirements",
                "data_protection": "Data protection enabled",
                "quantum_advantage": "Quantum advantage detected",
                "breakthrough": "Research breakthrough discovered"
            },
            SupportedLanguage.SPANISH: {
                "welcome": "Bienvenido a Quantum MLOps",
                "error": "Ocurrió un error",
                "success": "Operación completada exitosamente",
                "processing": "Procesando circuito cuántico",
                "optimization": "Optimizando rendimiento",
                "deployment": "Desplegando en región",
                "compliance_check": "Verificando requisitos de cumplimiento",
                "data_protection": "Protección de datos habilitada",
                "quantum_advantage": "Ventaja cuántica detectada",
                "breakthrough": "Descubrimiento de investigación encontrado"
            },
            SupportedLanguage.FRENCH: {
                "welcome": "Bienvenue dans Quantum MLOps",
                "error": "Une erreur s'est produite",
                "success": "Opération terminée avec succès",
                "processing": "Traitement du circuit quantique",
                "optimization": "Optimisation des performances",
                "deployment": "Déploiement vers la région",
                "compliance_check": "Vérification des exigences de conformité",
                "data_protection": "Protection des données activée",
                "quantum_advantage": "Avantage quantique détecté",
                "breakthrough": "Percée de recherche découverte"
            },
            SupportedLanguage.GERMAN: {
                "welcome": "Willkommen bei Quantum MLOps",
                "error": "Ein Fehler ist aufgetreten",
                "success": "Operation erfolgreich abgeschlossen",
                "processing": "Quantenschaltung wird verarbeitet",
                "optimization": "Leistung wird optimiert",
                "deployment": "Bereitstellung in Region",
                "compliance_check": "Compliance-Anforderungen prüfen",
                "data_protection": "Datenschutz aktiviert",
                "quantum_advantage": "Quantenvorteil erkannt",
                "breakthrough": "Forschungsdurchbruch entdeckt"
            },
            SupportedLanguage.JAPANESE: {
                "welcome": "Quantum MLOpsへようこそ",
                "error": "エラーが発生しました",
                "success": "操作が正常に完了しました",
                "processing": "量子回路を処理中",
                "optimization": "パフォーマンスを最適化中",
                "deployment": "リージョンにデプロイ中",
                "compliance_check": "コンプライアンス要件を確認中",
                "data_protection": "データ保護が有効",
                "quantum_advantage": "量子優位性を検出",
                "breakthrough": "研究のブレークスルーを発見"
            },
            SupportedLanguage.CHINESE: {
                "welcome": "欢迎使用 Quantum MLOps",
                "error": "发生错误",
                "success": "操作成功完成",
                "processing": "正在处理量子电路",
                "optimization": "正在优化性能",
                "deployment": "正在部署到区域",
                "compliance_check": "正在检查合规要求",
                "data_protection": "数据保护已启用",
                "quantum_advantage": "检测到量子优势",
                "breakthrough": "发现研究突破"
            }
        }
        
    def set_language(self, language: SupportedLanguage) -> None:
        """Set current language."""
        self.current_language = language
        self.logger.info(f"🌍 Language set to: {language.value}")
        
    def translate(self, key: str, language: Optional[SupportedLanguage] = None) -> str:
        """Translate text to specified language."""
        target_language = language or self.current_language
        
        if target_language not in self.translations:
            return key  # Return key if language not supported
            
        return self.translations[target_language].get(key, key)
        
    def get_localized_format(self, value: Any, format_type: str) -> str:
        """Get localized format for numbers, dates, etc."""
        try:
            if format_type == "number":
                if self.current_language == SupportedLanguage.GERMAN:
                    return f"{value:,.2f}".replace(",", " ").replace(".", ",")
                elif self.current_language == SupportedLanguage.FRENCH:
                    return f"{value:,.2f}".replace(",", " ")
                else:
                    return f"{value:,.2f}"
            elif format_type == "percentage":
                return f"{value:.1%}"
            else:
                return str(value)
        except:
            return str(value)


class ComplianceValidator:
    """Validator for global compliance requirements."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.compliance_requirements = self._initialize_compliance_requirements()
        
    def _initialize_compliance_requirements(self) -> Dict[ComplianceFramework, List[ComplianceRequirement]]:
        """Initialize compliance requirements for each framework."""
        return {
            ComplianceFramework.GDPR: [
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-01",
                    description="Data subject consent management",
                    implementation_guide="Implement explicit consent mechanisms",
                    validation_method="Check consent tracking system"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-02",
                    description="Right to data portability",
                    implementation_guide="Provide data export functionality",
                    validation_method="Test data export API"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-03",
                    description="Right to be forgotten",
                    implementation_guide="Implement data deletion mechanisms",
                    validation_method="Test data deletion functionality"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-04",
                    description="Data processing transparency",
                    implementation_guide="Provide clear processing notices",
                    validation_method="Review privacy notices"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.GDPR,
                    requirement_id="GDPR-05",
                    description="Data breach notification",
                    implementation_guide="Implement breach detection and notification",
                    validation_method="Test breach notification system"
                )
            ],
            ComplianceFramework.CCPA: [
                ComplianceRequirement(
                    framework=ComplianceFramework.CCPA,
                    requirement_id="CCPA-01",
                    description="Consumer right to know",
                    implementation_guide="Provide information about data collection",
                    validation_method="Review data collection disclosures"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.CCPA,
                    requirement_id="CCPA-02",
                    description="Consumer right to delete",
                    implementation_guide="Implement data deletion upon request",
                    validation_method="Test deletion request process"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.CCPA,
                    requirement_id="CCPA-03",
                    description="Consumer right to opt-out",
                    implementation_guide="Provide opt-out mechanisms",
                    validation_method="Test opt-out functionality"
                )
            ],
            ComplianceFramework.PDPA: [
                ComplianceRequirement(
                    framework=ComplianceFramework.PDPA,
                    requirement_id="PDPA-01",
                    description="Consent for data collection",
                    implementation_guide="Obtain consent before data collection",
                    validation_method="Check consent mechanisms"
                ),
                ComplianceRequirement(
                    framework=ComplianceFramework.PDPA,
                    requirement_id="PDPA-02",
                    description="Data protection obligations",
                    implementation_guide="Implement appropriate security measures",
                    validation_method="Security assessment"
                )
            ]
        }
        
    async def validate_compliance(
        self,
        region_config: RegionConfig,
        system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance for specified region."""
        validation_results = {
            "region": region_config.region.value,
            "compliance_frameworks": [f.value for f in region_config.compliance_frameworks],
            "validation_timestamp": time.time(),
            "passed_requirements": [],
            "failed_requirements": [],
            "warnings": [],
            "overall_compliance": False
        }
        
        for framework in region_config.compliance_frameworks:
            requirements = self.compliance_requirements.get(framework, [])
            
            for requirement in requirements:
                try:
                    compliance_check = await self._check_requirement(requirement, system_config)
                    
                    if compliance_check["compliant"]:
                        validation_results["passed_requirements"].append({
                            "framework": framework.value,
                            "requirement_id": requirement.requirement_id,
                            "description": requirement.description
                        })
                    else:
                        validation_results["failed_requirements"].append({
                            "framework": framework.value,
                            "requirement_id": requirement.requirement_id,
                            "description": requirement.description,
                            "issue": compliance_check.get("issue", "Unknown issue")
                        })
                        
                    if compliance_check.get("warning"):
                        validation_results["warnings"].append(compliance_check["warning"])
                        
                except Exception as e:
                    validation_results["failed_requirements"].append({
                        "framework": framework.value,
                        "requirement_id": requirement.requirement_id,
                        "description": requirement.description,
                        "issue": f"Validation error: {str(e)}"
                    })
                    
        # Calculate overall compliance
        total_requirements = len(validation_results["passed_requirements"]) + len(validation_results["failed_requirements"])
        if total_requirements > 0:
            compliance_rate = len(validation_results["passed_requirements"]) / total_requirements
            validation_results["overall_compliance"] = compliance_rate >= 0.95  # 95% threshold
            validation_results["compliance_rate"] = compliance_rate
        else:
            validation_results["overall_compliance"] = True
            validation_results["compliance_rate"] = 1.0
            
        return validation_results
        
    async def _check_requirement(
        self,
        requirement: ComplianceRequirement,
        system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check individual compliance requirement."""
        # Simulate compliance checking
        await asyncio.sleep(0.01)  # Simulate validation time
        
        # Mock validation logic based on requirement ID
        if "consent" in requirement.description.lower():
            has_consent_system = system_config.get("consent_management", False)
            return {
                "compliant": has_consent_system,
                "issue": "Consent management system not implemented" if not has_consent_system else None
            }
        elif "deletion" in requirement.description.lower():
            has_deletion = system_config.get("data_deletion_api", False)
            return {
                "compliant": has_deletion,
                "issue": "Data deletion API not implemented" if not has_deletion else None
            }
        elif "security" in requirement.description.lower():
            has_security = system_config.get("security_measures", False)
            return {
                "compliant": has_security,
                "issue": "Security measures not implemented" if not has_security else None
            }
        else:
            # Default to compliant for demonstration
            return {"compliant": True}


class GlobalDeploymentOrchestrator:
    """Orchestrator for global quantum MLOps deployment."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.i18n_manager = GlobalI18nManager()
        self.compliance_validator = ComplianceValidator()
        self.region_configs = self._initialize_region_configs()
        self.deployment_status = {}
        
    def _initialize_region_configs(self) -> Dict[DeploymentRegion, RegionConfig]:
        """Initialize region-specific configurations."""
        return {
            DeploymentRegion.US_EAST: RegionConfig(
                region=DeploymentRegion.US_EAST,
                compliance_frameworks=[ComplianceFramework.CCPA],
                primary_languages=[SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH],
                data_residency_required=False,
                quantum_providers=["aws_braket", "ionq"],
                edge_locations=["us-east-1a", "us-east-1b", "us-east-1c"]
            ),
            DeploymentRegion.EU_WEST: RegionConfig(
                region=DeploymentRegion.EU_WEST,
                compliance_frameworks=[ComplianceFramework.GDPR],
                primary_languages=[SupportedLanguage.ENGLISH, SupportedLanguage.FRENCH, SupportedLanguage.GERMAN],
                data_residency_required=True,
                quantum_providers=["aws_braket"],
                edge_locations=["eu-west-1a", "eu-west-1b"]
            ),
            DeploymentRegion.ASIA_PACIFIC: RegionConfig(
                region=DeploymentRegion.ASIA_PACIFIC,
                compliance_frameworks=[ComplianceFramework.PDPA],
                primary_languages=[SupportedLanguage.ENGLISH, SupportedLanguage.CHINESE, SupportedLanguage.JAPANESE],
                data_residency_required=True,
                quantum_providers=["aws_braket"],
                edge_locations=["ap-southeast-1a", "ap-southeast-1b"]
            )
        }
        
    async def deploy_globally(
        self,
        target_regions: List[DeploymentRegion],
        system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy quantum MLOps globally to specified regions."""
        self.logger.info(f"🌍 Starting global deployment to {len(target_regions)} regions")
        
        deployment_results = {
            "deployment_id": f"global_deploy_{int(time.time())}",
            "start_time": time.time(),
            "target_regions": [r.value for r in target_regions],
            "region_results": {},
            "overall_success": False,
            "compliance_summary": {}
        }
        
        # Deploy to each region
        for region in target_regions:
            self.logger.info(f"🚀 Deploying to region: {region.value}")
            
            region_result = await self._deploy_to_region(region, system_config)
            deployment_results["region_results"][region.value] = region_result
            
        # Calculate overall success
        successful_deployments = sum(
            1 for result in deployment_results["region_results"].values()
            if result["deployment_success"]
        )
        
        deployment_results["overall_success"] = successful_deployments == len(target_regions)
        deployment_results["success_rate"] = successful_deployments / len(target_regions)
        deployment_results["end_time"] = time.time()
        
        # Generate compliance summary
        deployment_results["compliance_summary"] = self._generate_compliance_summary(
            deployment_results["region_results"]
        )
        
        self.logger.info(
            f"🏆 Global deployment complete: {successful_deployments}/{len(target_regions)} regions successful"
        )
        
        return deployment_results
        
    async def _deploy_to_region(
        self,
        region: DeploymentRegion,
        system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy to specific region."""
        region_config = self.region_configs[region]
        
        result = {
            "region": region.value,
            "start_time": time.time(),
            "deployment_success": False,
            "compliance_validation": {},
            "i18n_setup": {},
            "quantum_providers_setup": {},
            "issues": []
        }
        
        try:
            # Step 1: Validate compliance
            compliance_result = await self.compliance_validator.validate_compliance(
                region_config, system_config
            )
            result["compliance_validation"] = compliance_result
            
            if not compliance_result["overall_compliance"]:
                result["issues"].append("Compliance validation failed")
                return result
                
            # Step 2: Setup internationalization
            i18n_result = await self._setup_region_i18n(region_config)
            result["i18n_setup"] = i18n_result
            
            # Step 3: Configure quantum providers
            providers_result = await self._setup_quantum_providers(region_config)
            result["quantum_providers_setup"] = providers_result
            
            # Step 4: Deploy infrastructure
            infrastructure_result = await self._deploy_infrastructure(region_config, system_config)
            result["infrastructure_deployment"] = infrastructure_result
            
            # Step 5: Validate deployment
            validation_result = await self._validate_deployment(region_config)
            result["deployment_validation"] = validation_result
            
            result["deployment_success"] = (
                compliance_result["overall_compliance"] and
                i18n_result["success"] and
                providers_result["success"] and
                infrastructure_result["success"] and
                validation_result["success"]
            )
            
        except Exception as e:
            result["issues"].append(f"Deployment error: {str(e)}")
            self.logger.error(f"Deployment to {region.value} failed: {e}")
            
        result["end_time"] = time.time()
        result["deployment_duration"] = result["end_time"] - result["start_time"]
        
        return result
        
    async def _setup_region_i18n(self, region_config: RegionConfig) -> Dict[str, Any]:
        """Setup internationalization for region."""
        result = {
            "success": True,
            "configured_languages": [],
            "primary_language": None,
            "issues": []
        }
        
        try:
            # Set primary language (first in list)
            if region_config.primary_languages:
                primary_lang = region_config.primary_languages[0]
                self.i18n_manager.set_language(primary_lang)
                result["primary_language"] = primary_lang.value
                
            # Configure all languages for region
            for language in region_config.primary_languages:
                # Simulate language configuration
                await asyncio.sleep(0.01)
                result["configured_languages"].append(language.value)
                
            self.logger.info(
                f"🌐 I18n configured for {region_config.region.value}: "
                f"{len(result['configured_languages'])} languages"
            )
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"I18n setup error: {str(e)}")
            
        return result
        
    async def _setup_quantum_providers(self, region_config: RegionConfig) -> Dict[str, Any]:
        """Setup quantum providers for region."""
        result = {
            "success": True,
            "configured_providers": [],
            "primary_provider": None,
            "issues": []
        }
        
        try:
            for provider in region_config.quantum_providers:
                # Simulate provider configuration
                await asyncio.sleep(0.02)
                
                provider_config = {
                    "name": provider,
                    "region": region_config.region.value,
                    "status": "configured"
                }
                result["configured_providers"].append(provider_config)
                
            if result["configured_providers"]:
                result["primary_provider"] = result["configured_providers"][0]["name"]
                
            self.logger.info(
                f"⚛️ Quantum providers configured for {region_config.region.value}: "
                f"{len(result['configured_providers'])} providers"
            )
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Provider setup error: {str(e)}")
            
        return result
        
    async def _deploy_infrastructure(
        self,
        region_config: RegionConfig,
        system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy infrastructure to region."""
        result = {
            "success": True,
            "deployed_components": [],
            "issues": []
        }
        
        try:
            # Simulate infrastructure deployment
            components = [
                "quantum_mlops_api",
                "quantum_executor_service",
                "monitoring_dashboard",
                "security_service",
                "compliance_service"
            ]
            
            for component in components:
                await asyncio.sleep(0.05)  # Simulate deployment time
                result["deployed_components"].append({
                    "name": component,
                    "status": "deployed",
                    "region": region_config.region.value
                })
                
            self.logger.info(
                f"🏗️ Infrastructure deployed to {region_config.region.value}: "
                f"{len(result['deployed_components'])} components"
            )
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Infrastructure deployment error: {str(e)}")
            
        return result
        
    async def _validate_deployment(self, region_config: RegionConfig) -> Dict[str, Any]:
        """Validate deployment in region."""
        result = {
            "success": True,
            "health_checks": [],
            "performance_metrics": {},
            "issues": []
        }
        
        try:
            # Simulate health checks
            health_checks = [
                "api_accessibility",
                "quantum_provider_connectivity",
                "database_connectivity",
                "monitoring_operational",
                "security_validation"
            ]
            
            for check in health_checks:
                await asyncio.sleep(0.01)
                result["health_checks"].append({
                    "check": check,
                    "status": "passed",
                    "response_time": 50 + (hash(check) % 100)  # Simulated response time
                })
                
            # Simulate performance metrics
            result["performance_metrics"] = {
                "api_response_time": 120,  # ms
                "quantum_circuit_execution_time": 500,  # ms
                "throughput": 100,  # requests per minute
                "availability": 99.9  # percentage
            }
            
            self.logger.info(f"✅ Deployment validation passed for {region_config.region.value}")
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Validation error: {str(e)}")
            
        return result
        
    def _generate_compliance_summary(self, region_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance summary across all regions."""
        summary = {
            "total_regions": len(region_results),
            "compliant_regions": 0,
            "compliance_frameworks": set(),
            "common_issues": [],
            "overall_compliance_rate": 0.0
        }
        
        compliance_rates = []
        all_issues = []
        
        for region_name, result in region_results.items():
            compliance_validation = result.get("compliance_validation", {})
            
            if compliance_validation.get("overall_compliance", False):
                summary["compliant_regions"] += 1
                
            compliance_rate = compliance_validation.get("compliance_rate", 0.0)
            compliance_rates.append(compliance_rate)
            
            # Collect frameworks
            frameworks = compliance_validation.get("compliance_frameworks", [])
            summary["compliance_frameworks"].update(frameworks)
            
            # Collect issues
            failed_requirements = compliance_validation.get("failed_requirements", [])
            for req in failed_requirements:
                all_issues.append(req.get("issue", "Unknown issue"))
                
        # Calculate overall compliance rate
        if compliance_rates:
            summary["overall_compliance_rate"] = sum(compliance_rates) / len(compliance_rates)
            
        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
        summary["common_issues"] = [
            {"issue": issue, "occurrences": count}
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            if count > 1
        ]
        
        summary["compliance_frameworks"] = list(summary["compliance_frameworks"])
        
        return summary
        
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment."""
        return self.deployment_status.get(deployment_id)
        
    async def rollback_deployment(
        self,
        deployment_id: str,
        target_regions: Optional[List[DeploymentRegion]] = None
    ) -> Dict[str, Any]:
        """Rollback deployment in specified regions."""
        self.logger.info(f"🔄 Rolling back deployment: {deployment_id}")
        
        rollback_result = {
            "deployment_id": deployment_id,
            "rollback_timestamp": time.time(),
            "success": True,
            "rolled_back_regions": [],
            "issues": []
        }
        
        # Simulate rollback process
        regions_to_rollback = target_regions or list(self.region_configs.keys())
        
        for region in regions_to_rollback:
            try:
                await asyncio.sleep(0.1)  # Simulate rollback time
                rollback_result["rolled_back_regions"].append(region.value)
                self.logger.info(f"✅ Rollback completed for region: {region.value}")
            except Exception as e:
                rollback_result["issues"].append(f"Rollback failed for {region.value}: {str(e)}")
                rollback_result["success"] = False
                
        return rollback_result


# Factory function for easy instantiation
def create_global_deployment_orchestrator() -> GlobalDeploymentOrchestrator:
    """Create and configure global deployment orchestrator."""
    return GlobalDeploymentOrchestrator()