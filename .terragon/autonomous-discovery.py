#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
For quantum-mlops-workbench (Advanced Maturity Repository)
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml
import re


@dataclass
class ValueItem:
    """Discovered value delivery item."""
    
    id: str
    title: str
    description: str
    category: str
    source: str
    files: List[str]
    estimated_effort: float  # hours
    impact_score: float     # 1-10
    confidence: float       # 0-1
    risk_level: float      # 0-1
    wsjf_score: float      # calculated
    ice_score: float       # calculated
    technical_debt_score: float
    composite_score: float
    quantum_specific: bool
    created_at: str
    tags: List[str]


class AutonomousDiscoveryEngine:
    """Advanced value discovery engine for quantum MLOps repositories."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize state
        self.discovered_items: List[ValueItem] = []
        self.execution_history: List[Dict] = []
        self.learning_data: Dict = {}
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Configuration not found at {self.config_path}")
            sys.exit(1)
    
    def discover_value_items(self) -> List[ValueItem]:
        """Comprehensive value discovery across all signal sources."""
        
        print("🔍 Starting autonomous value discovery...")
        items = []
        
        # 1. Git History Analysis
        items.extend(self._discover_from_git_history())
        
        # 2. Static Code Analysis  
        items.extend(self._discover_from_static_analysis())
        
        # 3. Security Scanning
        items.extend(self._discover_from_security_scan())
        
        # 4. Dependency Analysis
        items.extend(self._discover_from_dependencies())
        
        # 5. Quantum-Specific Analysis
        items.extend(self._discover_quantum_specific_items())
        
        # 6. Performance Analysis
        items.extend(self._discover_performance_opportunities())
        
        # 7. Documentation Gap Analysis
        items.extend(self._discover_documentation_gaps())
        
        # Calculate scores for all items
        scored_items = [self._calculate_composite_score(item) for item in items]
        
        # Filter and deduplicate
        filtered_items = self._filter_and_deduplicate(scored_items)
        
        print(f"✅ Discovered {len(filtered_items)} value items")
        return filtered_items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover items from git history analysis."""
        items = []
        
        try:
            # Search for TODO/FIXME patterns
            result = subprocess.run([
                "git", "log", "--oneline", "--grep=TODO\\|FIXME\\|HACK\\|XXX", 
                "-n", "50"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash = line.split()[0]
                    message = ' '.join(line.split()[1:])
                    
                    items.append(ValueItem(
                        id=f"git-{commit_hash[:8]}",
                        title=f"Address technical note: {message[:50]}...",
                        description=f"Technical note found in commit history: {message}",
                        category="technicalDebt",
                        source="gitHistory",
                        files=[],
                        estimated_effort=2.0,
                        impact_score=4.0,
                        confidence=0.6,
                        risk_level=0.3,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=15.0,
                        composite_score=0.0,
                        quantum_specific=False,
                        created_at=datetime.now().isoformat(),
                        tags=["technical-debt", "code-quality"]
                    ))
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static code analysis."""
        items = []
        
        # Run ruff to find code quality issues
        try:
            result = subprocess.run([
                "python", "-m", "ruff", "check", "src/", "--output-format=json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                
                # Group issues by file and rule
                issue_groups = {}
                for issue in ruff_issues[:10]:  # Limit to top 10
                    key = f"{issue['filename']}-{issue['code']}"
                    if key not in issue_groups:
                        issue_groups[key] = []
                    issue_groups[key].append(issue)
                
                for group_key, group_issues in issue_groups.items():
                    issue = group_issues[0]  # Representative issue
                    
                    items.append(ValueItem(
                        id=f"ruff-{hash(group_key) % 10000}",
                        title=f"Fix {issue['code']}: {issue['message'][:40]}...",
                        description=f"Code quality issue: {issue['message']}",
                        category="technicalDebt",
                        source="staticAnalysis",
                        files=[issue['filename']],
                        estimated_effort=1.0,
                        impact_score=3.0,
                        confidence=0.8,
                        risk_level=0.2,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=8.0,
                        composite_score=0.0,
                        quantum_specific='quantum' in issue['filename'].lower(),
                        created_at=datetime.now().isoformat(),
                        tags=["code-quality", "linting"]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_from_security_scan(self) -> List[ValueItem]:
        """Discover security-related value items."""
        items = []
        
        # Check for security vulnerabilities in dependencies
        try:
            result = subprocess.run([
                "python", "-m", "safety", "check", "--json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                
                for vuln in safety_data[:5]:  # Top 5 vulnerabilities
                    items.append(ValueItem(
                        id=f"sec-{vuln.get('id', hash(vuln['package']) % 10000)}",
                        title=f"Fix {vuln['package']} vulnerability",
                        description=f"Security vulnerability in {vuln['package']}: {vuln.get('advisory', 'See details')}",
                        category="security",
                        source="securityScanning",
                        files=["requirements.txt", "pyproject.toml"],
                        estimated_effort=1.5,
                        impact_score=8.0,
                        confidence=0.9,
                        risk_level=0.7,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        quantum_specific=False,
                        created_at=datetime.now().isoformat(),
                        tags=["security", "vulnerability", "dependencies"]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related improvements."""
        items = []
        
        # Check for outdated packages
        try:
            result = subprocess.run([
                "python", "-m", "pip", "list", "--outdated", "--format=json"
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                for pkg in outdated[:8]:  # Top 8 outdated packages
                    items.append(ValueItem(
                        id=f"dep-{pkg['name'].lower()}",
                        title=f"Update {pkg['name']} to {pkg['latest_version']}",
                        description=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        category="technicalDebt",
                        source="dependencyAnalysis",
                        files=["requirements.txt", "pyproject.toml"],
                        estimated_effort=0.5,
                        impact_score=3.0,
                        confidence=0.7,
                        risk_level=0.4,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=5.0,
                        composite_score=0.0,
                        quantum_specific=pkg['name'].lower() in ['pennylane', 'qiskit', 'cirq'],
                        created_at=datetime.now().isoformat(),
                        tags=["dependencies", "updates"]
                    ))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_quantum_specific_items(self) -> List[ValueItem]:
        """Discover quantum-specific optimization opportunities."""
        items = []
        
        # Analyze quantum circuit files
        quantum_files = list(self.repo_path.glob("**/*.py"))
        quantum_patterns = [
            r'@qml\.qnode',
            r'pennylane\.',
            r'qiskit\.',
            r'cirq\.',
            r'quantum_circuit',
            r'QuantumCircuit',
        ]
        
        for py_file in quantum_files:
            try:
                content = py_file.read_text()
                
                # Check for quantum patterns
                for pattern in quantum_patterns:
                    if re.search(pattern, content):
                        # Check for optimization opportunities
                        if 'depth' not in content.lower() and 'optimize' not in content.lower():
                            items.append(ValueItem(
                                id=f"qopt-{py_file.stem}",
                                title=f"Optimize quantum circuits in {py_file.name}",
                                description=f"Add circuit optimization and depth analysis to {py_file.name}",
                                category="quantumSpecific",
                                source="quantumCircuitAnalysis",
                                files=[str(py_file.relative_to(self.repo_path))],
                                estimated_effort=3.0,
                                impact_score=6.0,
                                confidence=0.7,
                                risk_level=0.3,
                                wsjf_score=0.0,
                                ice_score=0.0,
                                technical_debt_score=0.0,
                                composite_score=0.0,
                                quantum_specific=True,
                                created_at=datetime.now().isoformat(),
                                tags=["quantum", "optimization", "circuits"]
                            ))
                        break
                        
            except Exception:
                continue
                
        return items
    
    def _discover_performance_opportunities(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Look for performance bottlenecks
        perf_patterns = [
            (r'for.*in.*range\(.*\):', "Loop optimization opportunity"),
            (r'\.append\(.*\)', "Consider list comprehension"),
            (r'time\.sleep\(', "Async/await optimization opportunity"),
        ]
        
        for py_file in self.repo_path.glob("src/**/*.py"):
            try:
                content = py_file.read_text()
                
                for pattern, suggestion in perf_patterns:
                    matches = re.findall(pattern, content)
                    if len(matches) > 3:  # Multiple occurrences
                        items.append(ValueItem(
                            id=f"perf-{py_file.stem}-{hash(pattern) % 1000}",
                            title=f"Performance optimization in {py_file.name}",
                            description=f"{suggestion} - found {len(matches)} instances",
                            category="optimization",
                            source="performanceAnalysis",
                            files=[str(py_file.relative_to(self.repo_path))],
                            estimated_effort=4.0,
                            impact_score=5.0,
                            confidence=0.6,
                            risk_level=0.4,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            technical_debt_score=0.0,
                            composite_score=0.0,
                            quantum_specific='quantum' in str(py_file).lower(),
                            created_at=datetime.now().isoformat(),
                            tags=["performance", "optimization"]
                        ))
            except Exception:
                continue
                
        return items
    
    def _discover_documentation_gaps(self) -> List[ValueItem]:
        """Discover documentation improvement opportunities."""
        items = []
        
        # Check for missing docstrings
        for py_file in self.repo_path.glob("src/**/*.py"):
            try:
                content = py_file.read_text()
                
                # Count functions without docstrings
                func_pattern = r'def\s+\w+\([^)]*\):\s*\n(?!\s*""")'
                missing_docstrings = re.findall(func_pattern, content)
                
                if len(missing_docstrings) > 2:
                    items.append(ValueItem(
                        id=f"doc-{py_file.stem}",
                        title=f"Add missing docstrings in {py_file.name}",
                        description=f"Add docstrings to {len(missing_docstrings)} functions",
                        category="technicalDebt",
                        source="documentationAnalysis",
                        files=[str(py_file.relative_to(self.repo_path))],
                        estimated_effort=2.0,
                        impact_score=3.0,
                        confidence=0.8,
                        risk_level=0.1,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        technical_debt_score=6.0,
                        composite_score=0.0,
                        quantum_specific='quantum' in str(py_file).lower(),
                        created_at=datetime.now().isoformat(),
                        tags=["documentation", "docstrings"]
                    ))
            except Exception:
                continue
                
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> ValueItem:
        """Calculate comprehensive value score using WSJF + ICE + Technical Debt."""
        
        # WSJF Components
        user_business_value = item.impact_score * 0.4
        time_criticality = (10 - item.estimated_effort) * 0.3  # Shorter jobs score higher
        risk_reduction = (1 - item.risk_level) * 10 * 0.2
        opportunity_enablement = item.confidence * 10 * 0.1
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = max(item.estimated_effort, 0.5)  # Avoid division by zero
        
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE Score
        impact = item.impact_score
        confidence = item.confidence * 10
        ease = max(10 - item.estimated_effort, 1)  # Easier jobs score higher
        
        item.ice_score = impact * confidence * ease
        
        # Get weights from config
        weights = self.config['scoring']['weights']['advanced']
        
        # Composite Score
        normalized_wsjf = item.wsjf_score / 20  # Normalize to 0-1
        normalized_ice = item.ice_score / 1000  # Normalize to 0-1
        normalized_debt = item.technical_debt_score / 20  # Normalize to 0-1
        
        composite = (
            weights['wsjf'] * normalized_wsjf +
            weights['ice'] * normalized_ice +
            weights['technicalDebt'] * normalized_debt
        )
        
        # Apply category boosts
        category_config = self.config.get('taskCategories', {})
        if item.category in category_config:
            multiplier = category_config[item.category].get('scoreMultiplier', 1.0)
            composite *= multiplier
        
        # Apply quantum boost
        if item.quantum_specific:
            quantum_boost = self.config['scoring'].get('quantumBoost', 1.0)
            composite *= quantum_boost
        
        # Apply security boost
        if item.category == 'security':
            security_boost = self.config['scoring']['thresholds']['securityBoost']
            composite *= security_boost
        
        item.composite_score = round(composite * 100, 2)  # Scale to 0-100
        
        return item
    
    def _filter_and_deduplicate(self, items: List[ValueItem]) -> List[ValueItem]:
        """Filter items by score threshold and remove duplicates."""
        
        min_score = self.config['scoring']['thresholds']['minScore']
        
        # Filter by minimum score
        filtered = [item for item in items if item.composite_score >= min_score]
        
        # Deduplicate by title similarity
        deduplicated = []
        seen_titles = set()
        
        for item in sorted(filtered, key=lambda x: x.composite_score, reverse=True):
            # Simple deduplication by title similarity
            title_key = re.sub(r'[^a-zA-Z0-9]', '', item.title.lower())[:20]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                deduplicated.append(item)
        
        return deduplicated[:20]  # Limit to top 20 items
    
    def select_next_best_value_item(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item for execution."""
        
        if not items:
            return None
        
        # Sort by composite score
        sorted_items = sorted(items, key=lambda x: x.composite_score, reverse=True)
        
        max_risk = self.config['scoring']['thresholds']['maxRisk']
        
        for item in sorted_items:
            # Skip if risk exceeds threshold
            if item.risk_level > max_risk:
                continue
            
            # Skip if dependencies not met (placeholder logic)
            # In a real implementation, this would check for file locks, etc.
            
            return item
        
        return None
    
    def save_metrics(self, items: List[ValueItem]) -> None:
        """Save discovery metrics and results."""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_items_discovered": len(items),
            "items_by_category": {},
            "items_by_source": {},
            "average_score": 0.0,
            "quantum_specific_items": 0,
            "high_priority_items": 0,
            "discovered_items": [asdict(item) for item in items]
        }
        
        # Calculate category breakdown
        for item in items:
            metrics["items_by_category"][item.category] = metrics["items_by_category"].get(item.category, 0) + 1
            metrics["items_by_source"][item.source] = metrics["items_by_source"].get(item.source, 0) + 1
            
            if item.quantum_specific:
                metrics["quantum_specific_items"] += 1
            if item.composite_score > 50:
                metrics["high_priority_items"] += 1
        
        if items:
            metrics["average_score"] = sum(item.composite_score for item in items) / len(items)
        
        # Save to file
        os.makedirs(self.metrics_path.parent, exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def generate_backlog_markdown(self, items: List[ValueItem]) -> None:
        """Generate comprehensive backlog markdown file."""
        
        if not items:
            return
        
        next_item = self.select_next_best_value_item(items)
        
        content = f"""# 📊 Autonomous Value Backlog

**Repository**: quantum-mlops-workbench  
**Maturity Level**: Advanced (85-95%)  
**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Next Discovery**: {(datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🎯 Next Best Value Item

"""
        
        if next_item:
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.0f} | **Tech Debt**: {next_item.technical_debt_score:.0f}
- **Estimated Effort**: {next_item.estimated_effort} hours
- **Category**: {next_item.category}
- **Impact**: {next_item.impact_score}/10 | **Confidence**: {next_item.confidence*100:.0f}% | **Risk**: {next_item.risk_level*100:.0f}%
- **Quantum-Specific**: {'✅' if next_item.quantum_specific else '❌'}

**Description**: {next_item.description}

"""
        else:
            content += "No items currently meet the selection criteria.\n\n"
        
        content += """## 📋 Top Value Items

| Rank | ID | Title | Score | Category | Effort | Quantum |
|------|-----|--------|-------|----------|--------|---------|
"""
        
        for i, item in enumerate(sorted(items, key=lambda x: x.composite_score, reverse=True)[:15], 1):
            quantum_icon = "🔬" if item.quantum_specific else "⚙️"
            content += f"| {i} | {item.id.upper()} | {item.title[:40]}... | {item.composite_score} | {item.category} | {item.estimated_effort}h | {quantum_icon} |\n"
        
        # Add category breakdown
        content += f"""

## 📈 Discovery Metrics

- **Total Items Discovered**: {len(items)}
- **Average Score**: {sum(item.composite_score for item in items) / len(items):.1f}
- **Quantum-Specific Items**: {sum(1 for item in items if item.quantum_specific)}
- **High Priority Items (>50 score)**: {sum(1 for item in items if item.composite_score > 50)}

### By Category
"""
        
        category_counts = {}
        for item in items:
            category_counts[item.category] = category_counts.get(item.category, 0) + 1
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            content += f"- **{category.title()}**: {count} items\n"
        
        content += f"""

### By Source
"""
        
        source_counts = {}
        for item in items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
            
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            content += f"- **{source}**: {count} items\n"
        
        content += """

## 🔄 Continuous Discovery Stats

- **Discovery Frequency**: Hourly security scans, daily comprehensive analysis
- **Scoring Model**: WSJF + ICE + Technical Debt (adaptive weights)
- **Quality Gates**: 85% test coverage, <3% performance regression
- **Execution Constraints**: Max 1 concurrent task, 8 daily tasks

## 🎯 Focus Areas for Advanced Repositories

1. **Performance Optimization** - Algorithm and circuit efficiency improvements
2. **Quantum Hardware Integration** - Multi-provider compatibility and optimization  
3. **Security Hardening** - Vulnerability management and compliance
4. **Technical Debt Reduction** - Code quality and maintainability improvements
5. **Modernization** - Framework updates and architecture evolution

---

*🤖 Generated by Terragon Autonomous SDLC Engine*  
*Configuration: `.terragon/value-config.yaml`*  
*Metrics: `.terragon/value-metrics.json`*
"""
        
        # Write backlog file
        with open(self.backlog_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Generated backlog at {self.backlog_path}")


def main():
    """Main execution function."""
    
    print("🚀 Terragon Autonomous SDLC Value Discovery Engine")
    print("=" * 60)
    
    engine = AutonomousDiscoveryEngine()
    
    # Discover value items
    items = engine.discover_value_items()
    
    if not items:
        print("ℹ️ No actionable value items discovered")
        return
    
    # Select next best item
    next_item = engine.select_next_best_value_item(items)
    
    if next_item:
        print(f"\n🎯 Next Best Value Item:")
        print(f"   [{next_item.id.upper()}] {next_item.title}")
        print(f"   Score: {next_item.composite_score} | Effort: {next_item.estimated_effort}h")
        print(f"   Category: {next_item.category} | Quantum: {'Yes' if next_item.quantum_specific else 'No'}")
    
    # Save metrics and generate backlog
    engine.save_metrics(items)
    engine.generate_backlog_markdown(items)
    
    print(f"\n📊 Discovery Summary:")
    print(f"   Total Items: {len(items)}")
    print(f"   Average Score: {sum(item.composite_score for item in items) / len(items):.1f}")
    print(f"   Quantum Items: {sum(1 for item in items if item.quantum_specific)}")
    print(f"   High Priority: {sum(1 for item in items if item.composite_score > 50)}")
    
    print(f"\n✅ Autonomous discovery complete!")


if __name__ == "__main__":
    main()