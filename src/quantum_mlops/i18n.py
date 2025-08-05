"""Internationalization support for quantum MLOps workbench."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class I18nManager:
    """Internationalization manager for quantum MLOps."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH) -> None:
        """Initialize i18n manager.
        
        Args:
            default_language: Default language for translations
        """
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load translation files for all supported languages."""
        # Define translations inline for core functionality
        self.translations = {
            "en": {
                # Core messages
                "quantum_pipeline_initialized": "Quantum ML pipeline initialized",
                "training_started": "Training started with {epochs} epochs",
                "training_completed": "Training completed in {time:.2f}s",
                "model_saved": "Model saved to {path}",
                "model_loaded": "Model loaded from {path}",
                "evaluation_completed": "Evaluation completed - Accuracy: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "Invalid quantum circuit provided",
                "invalid_qubits": "Number of qubits must be positive",
                "backend_unavailable": "Quantum backend {backend} is unavailable",
                "optimization_failed": "Hyperparameter optimization failed",
                "compilation_error": "Circuit compilation failed for {hardware}",
                
                # Hardware messages
                "backend_connected": "Connected to quantum backend: {backend}",
                "backend_disconnected": "Disconnected from quantum backend",
                "hardware_queue_status": "Queue position: {position}, estimated wait: {time}",
                "shots_remaining": "Quantum shots remaining: {shots}",
                
                # Progress messages
                "epoch_progress": "Epoch {epoch}/{total}: Loss={loss:.4f}, Accuracy={accuracy:.4f}",
                "gradient_variance": "Gradient variance: {variance:.6f}",
                "circuit_depth": "Circuit depth: {depth}",
                "fidelity": "Quantum state fidelity: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "Checking quantum backend status...",
                "test_execution": "Executing quantum tests...",
                "benchmark_running": "Running benchmark comparison...",
                "optimization_progress": "Optimization progress: {progress}%",
                
                # Validation messages
                "parameter_validation": "Validating parameters...",
                "circuit_validation": "Validating quantum circuit...",
                "data_validation": "Validating input data...",
                "noise_analysis": "Performing noise analysis...",
                
                # Monitoring messages
                "monitoring_started": "Quantum monitoring started",
                "metrics_logged": "Metrics logged to {system}",
                "alert_triggered": "Alert triggered: {message}",
                "health_check": "System health check: {status}",
            },
            
            "es": {
                # Core messages
                "quantum_pipeline_initialized": "Pipeline cuántico ML inicializado",
                "training_started": "Entrenamiento iniciado con {epochs} épocas",
                "training_completed": "Entrenamiento completado en {time:.2f}s",
                "model_saved": "Modelo guardado en {path}",
                "model_loaded": "Modelo cargado desde {path}",
                "evaluation_completed": "Evaluación completada - Precisión: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "Circuito cuántico inválido proporcionado",
                "invalid_qubits": "El número de qubits debe ser positivo",
                "backend_unavailable": "Backend cuántico {backend} no disponible",
                "optimization_failed": "Optimización de hiperparámetros falló",
                "compilation_error": "Compilación del circuito falló para {hardware}",
                
                # Hardware messages
                "backend_connected": "Conectado al backend cuántico: {backend}",
                "backend_disconnected": "Desconectado del backend cuántico",
                "hardware_queue_status": "Posición en cola: {position}, espera estimada: {time}",
                "shots_remaining": "Disparos cuánticos restantes: {shots}",
                
                # Progress messages
                "epoch_progress": "Época {epoch}/{total}: Pérdida={loss:.4f}, Precisión={accuracy:.4f}",
                "gradient_variance": "Varianza del gradiente: {variance:.6f}",
                "circuit_depth": "Profundidad del circuito: {depth}",
                "fidelity": "Fidelidad del estado cuántico: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "Verificando estado del backend cuántico...",
                "test_execution": "Ejecutando pruebas cuánticas...",
                "benchmark_running": "Ejecutando comparación de benchmarks...",
                "optimization_progress": "Progreso de optimización: {progress}%",
                
                # Validation messages
                "parameter_validation": "Validando parámetros...",
                "circuit_validation": "Validando circuito cuántico...",
                "data_validation": "Validando datos de entrada...",
                "noise_analysis": "Realizando análisis de ruido...",
                
                # Monitoring messages
                "monitoring_started": "Monitoreo cuántico iniciado",
                "metrics_logged": "Métricas registradas en {system}",
                "alert_triggered": "Alerta activada: {message}",
                "health_check": "Verificación de salud del sistema: {status}",
            },
            
            "fr": {
                # Core messages
                "quantum_pipeline_initialized": "Pipeline quantique ML initialisé",
                "training_started": "Entraînement commencé avec {epochs} époques",
                "training_completed": "Entraînement terminé en {time:.2f}s",
                "model_saved": "Modèle sauvegardé dans {path}",
                "model_loaded": "Modèle chargé depuis {path}",
                "evaluation_completed": "Évaluation terminée - Précision: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "Circuit quantique invalide fourni",
                "invalid_qubits": "Le nombre de qubits doit être positif",
                "backend_unavailable": "Backend quantique {backend} indisponible",
                "optimization_failed": "Optimisation des hyperparamètres échouée",
                "compilation_error": "Compilation du circuit échouée pour {hardware}",
                
                # Hardware messages
                "backend_connected": "Connecté au backend quantique: {backend}",
                "backend_disconnected": "Déconnecté du backend quantique",
                "hardware_queue_status": "Position dans la file: {position}, attente estimée: {time}",
                "shots_remaining": "Tirs quantiques restants: {shots}",
                
                # Progress messages
                "epoch_progress": "Époque {epoch}/{total}: Perte={loss:.4f}, Précision={accuracy:.4f}",
                "gradient_variance": "Variance du gradient: {variance:.6f}",
                "circuit_depth": "Profondeur du circuit: {depth}",
                "fidelity": "Fidélité de l'état quantique: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "Vérification du statut du backend quantique...",
                "test_execution": "Exécution des tests quantiques...",
                "benchmark_running": "Exécution de la comparaison benchmark...",
                "optimization_progress": "Progrès d'optimisation: {progress}%",
                
                # Validation messages
                "parameter_validation": "Validation des paramètres...",
                "circuit_validation": "Validation du circuit quantique...",
                "data_validation": "Validation des données d'entrée...",
                "noise_analysis": "Analyse du bruit en cours...",
                
                # Monitoring messages
                "monitoring_started": "Surveillance quantique démarrée",
                "metrics_logged": "Métriques enregistrées dans {system}",
                "alert_triggered": "Alerte déclenchée: {message}",
                "health_check": "Vérification de la santé du système: {status}",
            },
            
            "de": {
                # Core messages
                "quantum_pipeline_initialized": "Quantum-ML-Pipeline initialisiert",
                "training_started": "Training mit {epochs} Epochen gestartet",
                "training_completed": "Training in {time:.2f}s abgeschlossen",
                "model_saved": "Modell in {path} gespeichert",
                "model_loaded": "Modell von {path} geladen",
                "evaluation_completed": "Bewertung abgeschlossen - Genauigkeit: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "Ungültiger Quantenschaltkreis bereitgestellt",
                "invalid_qubits": "Anzahl der Qubits muss positiv sein",
                "backend_unavailable": "Quantum-Backend {backend} nicht verfügbar",
                "optimization_failed": "Hyperparameter-Optimierung fehlgeschlagen",
                "compilation_error": "Schaltkreis-Kompilierung für {hardware} fehlgeschlagen",
                
                # Hardware messages
                "backend_connected": "Mit Quantum-Backend verbunden: {backend}",
                "backend_disconnected": "Von Quantum-Backend getrennt",
                "hardware_queue_status": "Warteschlangenposition: {position}, geschätzte Wartezeit: {time}",
                "shots_remaining": "Verbleibende Quantum-Shots: {shots}",
                
                # Progress messages
                "epoch_progress": "Epoche {epoch}/{total}: Verlust={loss:.4f}, Genauigkeit={accuracy:.4f}",
                "gradient_variance": "Gradienten-Varianz: {variance:.6f}",
                "circuit_depth": "Schaltkreis-Tiefe: {depth}",
                "fidelity": "Quantenzustands-Fidelität: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "Überprüfung des Quantum-Backend-Status...",
                "test_execution": "Ausführung von Quantentests...",
                "benchmark_running": "Benchmark-Vergleich läuft...",
                "optimization_progress": "Optimierungsfortschritt: {progress}%",
                
                # Validation messages
                "parameter_validation": "Parameter werden validiert...",
                "circuit_validation": "Quantenschaltkreis wird validiert...",
                "data_validation": "Eingabedaten werden validiert...",
                "noise_analysis": "Rauschanalyse wird durchgeführt...",
                
                # Monitoring messages
                "monitoring_started": "Quantum-Überwachung gestartet",
                "metrics_logged": "Metriken in {system} protokolliert",
                "alert_triggered": "Alarm ausgelöst: {message}",
                "health_check": "System-Gesundheitsprüfung: {status}",
            },
            
            "ja": {
                # Core messages
                "quantum_pipeline_initialized": "量子MLパイプラインが初期化されました",
                "training_started": "{epochs}エポックでトレーニングを開始しました",
                "training_completed": "トレーニングが{time:.2f}秒で完了しました",
                "model_saved": "モデルを{path}に保存しました",
                "model_loaded": "{path}からモデルを読み込みました",
                "evaluation_completed": "評価完了 - 精度: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "無効な量子回路が提供されました",
                "invalid_qubits": "量子ビット数は正の値である必要があります",
                "backend_unavailable": "量子バックエンド{backend}は利用できません",
                "optimization_failed": "ハイパーパラメータ最適化が失敗しました",
                "compilation_error": "{hardware}の回路コンパイルが失敗しました",
                
                # Hardware messages
                "backend_connected": "量子バックエンドに接続しました: {backend}",
                "backend_disconnected": "量子バックエンドから切断しました",
                "hardware_queue_status": "キュー位置: {position}, 推定待機時間: {time}",
                "shots_remaining": "残り量子ショット数: {shots}",
                
                # Progress messages
                "epoch_progress": "エポック {epoch}/{total}: 損失={loss:.4f}, 精度={accuracy:.4f}",
                "gradient_variance": "勾配分散: {variance:.6f}",
                "circuit_depth": "回路深度: {depth}",
                "fidelity": "量子状態忠実度: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "量子バックエンドステータスを確認中...",
                "test_execution": "量子テストを実行中...",
                "benchmark_running": "ベンチマーク比較を実行中...",
                "optimization_progress": "最適化進行状況: {progress}%",
                
                # Validation messages
                "parameter_validation": "パラメータを検証中...",
                "circuit_validation": "量子回路を検証中...",
                "data_validation": "入力データを検証中...",
                "noise_analysis": "ノイズ解析を実行中...",
                
                # Monitoring messages
                "monitoring_started": "量子モニタリングを開始しました",
                "metrics_logged": "{system}にメトリクスをログしました",
                "alert_triggered": "アラートが発生しました: {message}",
                "health_check": "システムヘルスチェック: {status}",
            },
            
            "zh": {
                # Core messages
                "quantum_pipeline_initialized": "量子机器学习管道已初始化",
                "training_started": "开始训练，共{epochs}个周期",
                "training_completed": "训练在{time:.2f}秒内完成",
                "model_saved": "模型已保存到{path}",
                "model_loaded": "已从{path}加载模型",
                "evaluation_completed": "评估完成 - 准确率: {accuracy:.4f}",
                
                # Error messages
                "invalid_circuit": "提供了无效的量子电路",
                "invalid_qubits": "量子比特数必须为正数",
                "backend_unavailable": "量子后端{backend}不可用",
                "optimization_failed": "超参数优化失败",
                "compilation_error": "{hardware}的电路编译失败",
                
                # Hardware messages
                "backend_connected": "已连接到量子后端: {backend}",
                "backend_disconnected": "已断开量子后端连接",
                "hardware_queue_status": "队列位置: {position}，预计等待时间: {time}",
                "shots_remaining": "剩余量子射击次数: {shots}",
                
                # Progress messages
                "epoch_progress": "周期 {epoch}/{total}: 损失={loss:.4f}, 准确率={accuracy:.4f}",
                "gradient_variance": "梯度方差: {variance:.6f}",
                "circuit_depth": "电路深度: {depth}",
                "fidelity": "量子态保真度: {fidelity:.4f}",
                
                # CLI messages
                "status_check": "正在检查量子后端状态...",
                "test_execution": "正在执行量子测试...",
                "benchmark_running": "正在运行基准比较...",
                "optimization_progress": "优化进度: {progress}%",
                
                # Validation messages
                "parameter_validation": "正在验证参数...",
                "circuit_validation": "正在验证量子电路...",
                "data_validation": "正在验证输入数据...",
                "noise_analysis": "正在执行噪声分析...",
                
                # Monitoring messages
                "monitoring_started": "量子监控已启动",
                "metrics_logged": "指标已记录到{system}",
                "alert_triggered": "触发警报: {message}",
                "health_check": "系统健康检查: {status}",
            }
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set the current language for translations.
        
        Args:
            language: Language to use for translations
        """
        self.current_language = language
        logger.info(f"Language set to: {language.value}")
    
    def get_language(self) -> SupportedLanguage:
        """Get the current language.
        
        Returns:
            Current language setting
        """
        return self.current_language
    
    def translate(self, key: str, **kwargs: Any) -> str:
        """Translate a message key to the current language.
        
        Args:
            key: Translation key
            **kwargs: Format arguments for the translated string
            
        Returns:
            Translated and formatted string
        """
        lang_code = self.current_language.value
        
        # Get translation from current language, fallback to English
        if lang_code in self.translations and key in self.translations[lang_code]:
            template = self.translations[lang_code][key]
        elif key in self.translations["en"]:
            template = self.translations["en"][key]
            logger.warning(f"Translation not found for key '{key}' in language '{lang_code}', using English")
        else:
            logger.error(f"Translation key '{key}' not found in any language")
            return key  # Return the key itself as fallback
        
        # Format the template with provided arguments
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing format argument {e} for translation key '{key}'")
            return template
        except Exception as e:
            logger.error(f"Error formatting translation for key '{key}': {e}")
            return template
    
    def get_available_languages(self) -> list[SupportedLanguage]:
        """Get list of available languages.
        
        Returns:
            List of supported languages
        """
        return list(SupportedLanguage)
    
    def get_language_name(self, language: SupportedLanguage) -> str:
        """Get the native name of a language.
        
        Args:
            language: Language to get name for
            
        Returns:
            Native name of the language
        """
        names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE: "中文"
        }
        return names.get(language, language.value)
    
    def add_translation(self, language: SupportedLanguage, key: str, value: str) -> None:
        """Add or update a translation.
        
        Args:
            language: Language for the translation
            key: Translation key
            value: Translation value
        """
        lang_code = language.value
        
        if lang_code not in self.translations:
            self.translations[lang_code] = {}
        
        self.translations[lang_code][key] = value
        logger.debug(f"Added translation for '{key}' in language '{lang_code}'")
    
    def load_custom_translations(self, file_path: str) -> None:
        """Load custom translations from a JSON file.
        
        Args:
            file_path: Path to JSON file containing translations
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_translations = json.load(f)
            
            # Merge custom translations with existing ones
            for lang_code, translations in custom_translations.items():
                if lang_code not in self.translations:
                    self.translations[lang_code] = {}
                
                self.translations[lang_code].update(translations)
            
            logger.info(f"Loaded custom translations from {file_path}")
            
        except FileNotFoundError:
            logger.warning(f"Custom translation file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing custom translation file: {e}")
        except Exception as e:
            logger.error(f"Error loading custom translations: {e}")
    
    def export_translations(self, file_path: str, language: Optional[SupportedLanguage] = None) -> None:
        """Export translations to a JSON file.
        
        Args:
            file_path: Path to save translations to
            language: Specific language to export, or None for all languages
        """
        try:
            if language:
                lang_code = language.value
                export_data = {lang_code: self.translations.get(lang_code, {})}
            else:
                export_data = self.translations
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported translations to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting translations: {e}")


# Global i18n manager instance
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager() -> I18nManager:
    """Get the global i18n manager instance.
    
    Returns:
        Global I18nManager instance
    """
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def set_language(language: SupportedLanguage) -> None:
    """Set the global language for translations.
    
    Args:
        language: Language to use for translations
    """
    get_i18n_manager().set_language(language)


def translate(key: str, **kwargs: Any) -> str:
    """Translate a message key using the global i18n manager.
    
    Args:
        key: Translation key
        **kwargs: Format arguments for the translated string
        
    Returns:
        Translated and formatted string
    """
    return get_i18n_manager().translate(key, **kwargs)


def get_current_language() -> SupportedLanguage:
    """Get the current global language setting.
    
    Returns:
        Current language setting
    """
    return get_i18n_manager().get_language()


# Convenience function for shorter import
_ = translate