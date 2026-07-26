"""
Security Metrics - Prometheus metrics for security monitoring

Features:
- Security event tracking
- Authentication failure monitoring
- Suspicious activity detection
- API key usage monitoring
- Rate limit tracking
"""

from typing import Optional, Dict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Security metrics disabled.")


class SecurityEventType(Enum):
    """Types of security events."""
    AUTH_FAILURE = "auth_failure"
    AUTH_SUCCESS = "auth_success"
    API_KEY_ROTATION = "api_key_rotation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INTEGRATION_CHECK = "integration_check"
    KEY_COMPROMISE = "key_compromise"
    ACCESS_DENIED = "access_denied"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"


@dataclass
class SecurityEvent:
    """Security event data."""
    event_type: SecurityEventType
    severity: str  # low, medium, high, critical
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    service: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecurityMetrics:
    """
    Prometheus metrics for security monitoring.

    Metrics tracked:
    - Security events by type and severity
    - Authentication failures by endpoint
    - API key usage
    - Rate limit violations
    - Suspicious activity patterns
    - Integration check status
    """

    def __init__(self):
        """Initialize security metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Event counters
        self.security_events_total = Counter(
            'security_events_total',
            'Total security events',
            ['event_type', 'severity', 'service']
        )

        # Authentication metrics
        self.auth_failures_total = Counter(
            'auth_failures_total',
            'Total authentication failures',
            ['endpoint', 'user_id', 'failure_reason']
        )

        self.auth_success_total = Counter(
            'auth_success_total',
            'Total successful authentications',
            ['endpoint', 'user_id']
        )

        # API key metrics
        self.api_key_usage_total = Counter(
            'api_key_usage_total',
            'Total API key usage',
            ['service', 'operation']
        )

        self.api_key_rotations_total = Counter(
            'api_key_rotations_total',
            'Total API key rotations',
            ['service', 'rotation_reason']
        )

        # Rate limiting metrics
        self.rate_limit_exceeded_total = Counter(
            'rate_limit_exceeded_total',
            'Total rate limit violations',
            ['endpoint', 'user_id', 'limit_type']
        )

        # Suspicious activity metrics
        self.suspicious_activity_total = Counter(
            'suspicious_activity_total',
            'Total suspicious activity detected',
            ['activity_type', 'severity', 'ip_address']
        )

        # Access control metrics
        self.access_denied_total = Counter(
            'access_denied_total',
            'Total access denied events',
            ['resource_type', 'resource_id', 'reason']
        )

        self.permission_denied_total = Counter(
            'permission_denied_total',
            'Total permission denied events',
            ['action', 'resource_type', 'role']
        )

        # Integration check metrics
        self.integration_check_total = Counter(
            'integration_check_total',
            'Total integration checks',
            ['integration', 'status']
        )

        self.integration_check_duration_seconds = Histogram(
            'integration_check_duration_seconds',
            'Duration of integration checks',
            ['integration']
        )

        # Data access metrics
        self.data_access_total = Counter(
            'data_access_total',
            'Total data access events',
            ['data_type', 'operation', 'user_id']
        )

        # Active security alerts gauge
        self.active_security_alerts = Gauge(
            'active_security_alerts',
            'Number of active security alerts',
            ['severity']
        )

        logger.info("Security metrics initialized")

    def track_event(self, event: SecurityEvent) -> None:
        """
        Track a security event.

        Args:
            event: SecurityEvent to track
        """
        if not PROMETHEUS_AVAILABLE:
            return

        labels = {
            'event_type': event.event_type.value,
            'severity': event.severity,
            'service': event.service or 'unknown'
        }

        self.security_events_total.labels(**labels).inc()

        # Log critical events
        if event.severity in ['high', 'critical']:
            logger.critical(
                f"Security event: {event.event_type.value} - {event.metadata}",
                extra={
                    'event_type': event.event_type.value,
                    'severity': event.severity,
                    'user_id': event.user_id,
                    'ip_address': event.ip_address,
                    'metadata': event.metadata
                }
            )

    def track_auth_failure(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        failure_reason: str = "unknown"
    ) -> None:
        """
        Track authentication failure.

        Args:
            endpoint: API endpoint
            user_id: User ID (if available)
            failure_reason: Reason for failure
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.auth_failures_total.labels(
            endpoint=endpoint,
            user_id=user_id or 'anonymous',
            failure_reason=failure_reason
        ).inc()

        self.security_events_total.labels(
            event_type=SecurityEventType.AUTH_FAILURE.value,
            severity='medium',
            service='auth'
        ).inc()

        logger.warning(
            f"Authentication failure: {endpoint} - {failure_reason}",
            extra={'user_id': user_id, 'endpoint': endpoint}
        )

    def track_auth_success(
        self,
        endpoint: str,
        user_id: str
    ) -> None:
        """
        Track successful authentication.

        Args:
            endpoint: API endpoint
            user_id: User ID
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.auth_success_total.labels(
            endpoint=endpoint,
            user_id=user_id
        ).inc()

    def track_api_key_usage(
        self,
        service: str,
        operation: str
    ) -> None:
        """
        Track API key usage.

        Args:
            service: Service name (e.g., 'binance', 'coinbase')
            operation: Operation type (e.g., 'trade', 'get_balance')
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.api_key_usage_total.labels(
            service=service,
            operation=operation
        ).inc()

    def track_api_key_rotation(
        self,
        service: str,
        reason: str = "scheduled"
    ) -> None:
        """
        Track API key rotation.

        Args:
            service: Service name
            reason: Rotation reason
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.api_key_rotations_total.labels(
            service=service,
            rotation_reason=reason
        ).inc()

        self.track_event(SecurityEvent(
            event_type=SecurityEventType.API_KEY_ROTATION,
            severity='low',
            service=service,
            metadata={'reason': reason}
        ))

    def track_rate_limit_exceeded(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        limit_type: str = "requests_per_minute"
    ) -> None:
        """
        Track rate limit violation.

        Args:
            endpoint: API endpoint
            user_id: User ID (if available)
            limit_type: Type of limit exceeded
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.rate_limit_exceeded_total.labels(
            endpoint=endpoint,
            user_id=user_id or 'anonymous',
            limit_type=limit_type
        ).inc()

        self.track_event(SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity='medium',
            endpoint=endpoint,
            user_id=user_id,
            metadata={'limit_type': limit_type}
        ))

    def track_suspicious_activity(
        self,
        activity_type: str,
        severity: str = "medium",
        ip_address: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Track suspicious activity.

        Args:
            activity_type: Type of suspicious activity
            severity: Severity level
            ip_address: IP address (if available)
            metadata: Additional metadata
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.suspicious_activity_total.labels(
            activity_type=activity_type,
            severity=severity,
            ip_address=ip_address or 'unknown'
        ).inc()

        self.track_event(SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            severity=severity,
            ip_address=ip_address,
            metadata=metadata or {}
        ))

        # Increment active alerts for high/critical severity
        if severity in ['high', 'critical']:
            self.active_security_alerts.labels(severity=severity).inc()

    def track_access_denied(
        self,
        resource_type: str,
        resource_id: str,
        reason: str = "unauthorized"
    ) -> None:
        """
        Track access denied event.

        Args:
            resource_type: Type of resource (e.g., 'database', 'api')
            resource_id: ID of resource
            reason: Reason for denial
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.access_denied_total.labels(
            resource_type=resource_type,
            resource_id=resource_id,
            reason=reason
        ).inc()

        self.track_event(SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED,
            severity='medium',
            metadata={
                'resource_type': resource_type,
                'resource_id': resource_id,
                'reason': reason
            }
        ))

    def track_permission_denied(
        self,
        action: str,
        resource_type: str,
        role: str
    ) -> None:
        """
        Track permission denied event.

        Args:
            action: Action attempted
            resource_type: Type of resource
            role: User role
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.permission_denied_total.labels(
            action=action,
            resource_type=resource_type,
            role=role
        ).inc()

    def track_integration_check(
        self,
        integration: str,
        status: str,
        duration_seconds: float
    ) -> None:
        """
        Track integration check.

        Args:
            integration: Integration name (e.g., 'database', 'redis', 'binance')
            status: Check status (success, failure)
            duration_seconds: Check duration
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.integration_check_total.labels(
            integration=integration,
            status=status
        ).inc()

        self.integration_check_duration_seconds.labels(
            integration=integration
        ).observe(duration_seconds)

        if status == 'failure':
            self.track_event(SecurityEvent(
                event_type=SecurityEventType.INTEGRATION_CHECK,
                severity='medium',
                service=integration,
                metadata={'status': status, 'duration': duration_seconds}
            ))

    def track_data_access(
        self,
        data_type: str,
        operation: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Track data access event.

        Args:
            data_type: Type of data (e.g., 'trades', 'positions', 'users')
            operation: Operation (read, write, delete)
            user_id: User ID
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.data_access_total.labels(
            data_type=data_type,
            operation=operation,
            user_id=user_id or 'system'
        ).inc()

        # Track sensitive data access as security event
        if data_type in ['users', 'api_keys', 'credentials']:
            self.track_event(SecurityEvent(
                event_type=SecurityEventType.DATA_ACCESS,
                severity='low',
                user_id=user_id,
                metadata={'data_type': data_type, 'operation': operation}
            ))

    def set_active_alerts(self, severity: str, count: int) -> None:
        """
        Set number of active alerts by severity.

        Args:
            severity: Severity level
            count: Number of alerts
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.active_security_alerts.labels(severity=severity).set(count)


# Singleton instance
_metrics: Optional[SecurityMetrics] = None


def get_security_metrics() -> SecurityMetrics:
    """Get or create global security metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = SecurityMetrics()
    return _metrics


# Decorator for tracking function calls with security metrics
def track_security_event(
    event_type: SecurityEventType,
    severity: str = "low"
):
    """
    Decorator to track security events on function calls.

    Usage:
        @track_security_event(SecurityEventType.AUTH_FAILURE, severity="medium")
        def authenticate_user(username, password):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_security_metrics()

            try:
                result = func(*args, **kwargs)

                # Track success
                if event_type == SecurityEventType.AUTH_SUCCESS:
                    metrics.track_auth_success(
                        endpoint=func.__name__,
                        user_id=kwargs.get('user_id')
                    )

                return result
            except Exception as e:
                # Track failure
                if event_type == SecurityEventType.AUTH_FAILURE:
                    metrics.track_auth_failure(
                        endpoint=func.__name__,
                        failure_reason=str(e)
                    )
                raise

        return wrapper
    return decorator


# Decorator for tracking rate limit violations
def track_rate_limit():
    """
    Decorator to track rate limit violations.

    Usage:
        @track_rate_limit()
        async def api_endpoint():
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            metrics = get_security_metrics()

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    metrics.track_rate_limit_exceeded(
                        endpoint=func.__name__,
                        limit_type="requests_per_minute"
                    )
                raise

        def sync_wrapper(*args, **kwargs):
            metrics = get_security_metrics()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower():
                    metrics.track_rate_limit_exceeded(
                        endpoint=func.__name__,
                        limit_type="requests_per_minute"
                    )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    import time

    # Get metrics instance
    metrics = SecurityMetrics()

    # Track authentication failure
    metrics.track_auth_failure(
        endpoint="/api/auth",
        user_id="user123",
        failure_reason="invalid_password"
    )

    # Track API key usage
    metrics.track_api_key_usage(
        service="binance",
        operation="place_order"
    )

    # Track suspicious activity
    metrics.track_suspicious_activity(
        activity_type="unusual_ip",
        severity="high",
        ip_address="192.168.1.100",
        metadata={"previous_ips": ["10.0.0.1"]}
    )

    # Track integration check
    metrics.track_integration_check(
        integration="database",
        status="success",
        duration_seconds=0.5
    )

    # Use decorator
    @track_security_event(SecurityEventType.AUTH_FAILURE, severity="medium")
    def login(username, password):
        if username != "admin" or password != "secret":
            raise ValueError("Invalid credentials")
        return True

    # Test decorated function
    try:
        login("admin", "wrong")
    except ValueError:
        pass

    print("Security metrics example completed")
