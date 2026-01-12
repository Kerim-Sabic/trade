"""Dashboard server for monitoring the trading AI system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import threading
from loguru import logger

from cryptoai.monitoring.metrics_collector import MetricsCollector
from cryptoai.monitoring.drift_detector import DriftDetector
from cryptoai.monitoring.alerting import AlertManager


@dataclass
class DashboardConfig:
    """Configuration for dashboard server."""
    host: str = "0.0.0.0"
    port: int = 8081
    refresh_interval_ms: int = 5000
    enable_websocket: bool = True


class DashboardServer:
    """
    Web dashboard for monitoring the trading AI system.

    Provides:
    - Real-time metrics visualization
    - Alert management
    - Model performance tracking
    - Drift detection status
    - System health overview
    """

    def __init__(
        self,
        metrics: MetricsCollector,
        drift_detector: DriftDetector,
        alert_manager: AlertManager,
        config: DashboardConfig,
    ):
        self.metrics = metrics
        self.drift_detector = drift_detector
        self.alert_manager = alert_manager
        self.config = config

        self._app = None
        self._server_thread: Optional[threading.Thread] = None

    def _setup_routes(self) -> None:
        """Setup API routes for dashboard."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            import uvicorn

            self._app = FastAPI(title="CryptoAI Trading Dashboard")

            # API endpoints
            @self._app.get("/api/health")
            async def health():
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            @self._app.get("/api/metrics")
            async def get_metrics():
                return self.metrics.get_all_metrics()

            @self._app.get("/api/metrics/{metric_name}")
            async def get_metric(metric_name: str):
                return self.metrics.get_metric(metric_name)

            @self._app.get("/api/metrics/{metric_name}/timeseries")
            async def get_timeseries(metric_name: str, hours: int = 1):
                return self.metrics.get_time_series(metric_name, hours=hours)

            @self._app.get("/api/drift")
            async def get_drift():
                return self.drift_detector.get_drift_summary()

            @self._app.get("/api/alerts")
            async def get_alerts():
                return [
                    {
                        "id": a.alert_id,
                        "level": a.level.value,
                        "title": a.title,
                        "message": a.message,
                        "source": a.source,
                        "timestamp": a.timestamp.isoformat(),
                        "acknowledged": a.acknowledged,
                        "resolved": a.resolved,
                    }
                    for a in self.alert_manager.get_active_alerts()
                ]

            @self._app.post("/api/alerts/{alert_id}/acknowledge")
            async def acknowledge_alert(alert_id: str):
                success = self.alert_manager.acknowledge(alert_id)
                return {"success": success}

            @self._app.post("/api/alerts/{alert_id}/resolve")
            async def resolve_alert(alert_id: str):
                success = self.alert_manager.resolve(alert_id)
                return {"success": success}

            @self._app.get("/api/prometheus")
            async def prometheus_metrics():
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(
                    self.metrics.export_prometheus(),
                    media_type="text/plain",
                )

            # Dashboard HTML
            @self._app.get("/", response_class=HTMLResponse)
            async def dashboard():
                return self._generate_dashboard_html()

            # WebSocket for real-time updates
            @self._app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                try:
                    while True:
                        data = {
                            "metrics": self.metrics.get_all_metrics(),
                            "drift": self.drift_detector.get_drift_summary(),
                            "alerts": [
                                {
                                    "id": a.alert_id,
                                    "level": a.level.value,
                                    "title": a.title,
                                }
                                for a in self.alert_manager.get_active_alerts()[:5]
                            ],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        await websocket.send_json(data)
                        await asyncio.sleep(self.config.refresh_interval_ms / 1000)
                except WebSocketDisconnect:
                    pass

            self._uvicorn = uvicorn
            logger.info("Dashboard routes configured")

        except ImportError as e:
            logger.warning(f"FastAPI not installed, dashboard disabled: {e}")

    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CryptoAI Trading Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8">CryptoAI Trading Dashboard</h1>

        <!-- Status Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-gray-400 text-sm">System Status</h3>
                <p id="system-status" class="text-2xl font-bold text-green-500">Healthy</p>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-gray-400 text-sm">Active Alerts</h3>
                <p id="alert-count" class="text-2xl font-bold text-yellow-500">0</p>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-gray-400 text-sm">Drift Detected</h3>
                <p id="drift-status" class="text-2xl font-bold">No</p>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-gray-400 text-sm">Inference/sec</h3>
                <p id="inference-rate" class="text-2xl font-bold">0</p>
            </div>
        </div>

        <!-- Charts -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-lg font-semibold mb-4">PnL Over Time</h3>
                <canvas id="pnl-chart"></canvas>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <h3 class="text-lg font-semibold mb-4">Inference Latency</h3>
                <canvas id="latency-chart"></canvas>
            </div>
        </div>

        <!-- Alerts Table -->
        <div class="bg-gray-800 rounded-lg p-4 mb-8">
            <h3 class="text-lg font-semibold mb-4">Active Alerts</h3>
            <table class="w-full">
                <thead>
                    <tr class="text-left text-gray-400">
                        <th class="pb-2">Level</th>
                        <th class="pb-2">Title</th>
                        <th class="pb-2">Source</th>
                        <th class="pb-2">Time</th>
                        <th class="pb-2">Actions</th>
                    </tr>
                </thead>
                <tbody id="alerts-table">
                </tbody>
            </table>
        </div>

        <!-- Drift Detection -->
        <div class="bg-gray-800 rounded-lg p-4">
            <h3 class="text-lg font-semibold mb-4">Drift Detection</h3>
            <div id="drift-details" class="grid grid-cols-1 md:grid-cols-3 gap-4">
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        let pnlChart, latencyChart;
        const pnlData = [];
        const latencyData = [];

        // Initialize charts
        function initCharts() {
            const pnlCtx = document.getElementById('pnl-chart').getContext('2d');
            pnlChart = new Chart(pnlCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'PnL',
                        data: [],
                        borderColor: 'rgb(34, 197, 94)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });

            const latencyCtx = document.getElementById('latency-chart').getContext('2d');
            latencyChart = new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [],
                        borderColor: 'rgb(59, 130, 246)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        }

        // Update dashboard
        function updateDashboard(data) {
            // Update status cards
            document.getElementById('alert-count').textContent = data.alerts.length;

            const driftDetected = data.drift.requires_retraining;
            const driftStatus = document.getElementById('drift-status');
            driftStatus.textContent = driftDetected ? 'Yes' : 'No';
            driftStatus.className = driftDetected ?
                'text-2xl font-bold text-red-500' :
                'text-2xl font-bold text-green-500';

            // Update alerts table
            const alertsTable = document.getElementById('alerts-table');
            alertsTable.innerHTML = data.alerts.map(alert => `
                <tr class="border-t border-gray-700">
                    <td class="py-2">
                        <span class="px-2 py-1 rounded text-xs ${
                            alert.level === 'critical' ? 'bg-red-600' :
                            alert.level === 'warning' ? 'bg-yellow-600' :
                            'bg-blue-600'
                        }">${alert.level}</span>
                    </td>
                    <td class="py-2">${alert.title}</td>
                    <td class="py-2 text-gray-400">${alert.id.split('_')[0]}</td>
                    <td class="py-2 text-gray-400">Just now</td>
                    <td class="py-2">
                        <button class="text-blue-400 hover:text-blue-300 mr-2"
                                onclick="acknowledgeAlert('${alert.id}')">Ack</button>
                        <button class="text-green-400 hover:text-green-300"
                                onclick="resolveAlert('${alert.id}')">Resolve</button>
                    </td>
                </tr>
            `).join('');

            // Update drift details
            const driftDetails = document.getElementById('drift-details');
            driftDetails.innerHTML = Object.entries(data.drift)
                .filter(([key]) => key !== 'requires_retraining')
                .map(([key, value]) => `
                    <div class="bg-gray-700 rounded p-3">
                        <h4 class="text-gray-400 text-sm">${key.replace('_', ' ')}</h4>
                        <p class="text-lg ${value.detected ? 'text-red-400' : 'text-green-400'}">
                            ${value.detected ? 'Detected' : 'Normal'}
                        </p>
                        <p class="text-gray-500 text-sm">Score: ${value.score?.toFixed(3) || 'N/A'}</p>
                    </div>
                `).join('');
        }

        // Alert actions
        async function acknowledgeAlert(alertId) {
            await fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
        }

        async function resolveAlert(alertId) {
            await fetch(`/api/alerts/${alertId}/resolve`, { method: 'POST' });
        }

        // WebSocket handlers
        ws.onopen = () => {
            console.log('Connected to dashboard');
            initCharts();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    </script>
</body>
</html>
        """

    def start(self) -> None:
        """Start dashboard server."""
        self._setup_routes()

        if self._app and hasattr(self, '_uvicorn'):
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
            )
            self._server_thread.start()
            logger.info(f"Dashboard started at http://{self.config.host}:{self.config.port}")

    def _run_server(self) -> None:
        """Run the server."""
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())

        self._uvicorn.run(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )

    def stop(self) -> None:
        """Stop dashboard server."""
        logger.info("Dashboard stopped")
