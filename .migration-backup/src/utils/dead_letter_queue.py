"""
Dead Letter Queue

Handles operations that fail after all retry attempts.
Stores failed operations for later inspection, manual retry, or alerting.
"""
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    """Failure reason categories."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"


class DeadLetterQueue:
    """
    Dead letter queue for failed operations.

    Features:
    - Persistent storage of failed operations
    - Categorization by operation type
    - Retry tracking
    - Manual retry capability
    - Automatic cleanup of old entries
    """

    def __init__(self, storage_path: str = "/tmp/dlq"):
        """
        Initialize dead letter queue.

        Args:
            storage_path: Path to store failed operations
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dead letter queue initialized: {storage_path}")

    def add(
        self,
        operation_type: str,
        operation_data: Dict,
        error: Exception,
        failure_reason: FailureReason = FailureReason.UNKNOWN,
        retry_count: int = 0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add failed operation to queue.

        Args:
            operation_type: Type of operation (e.g., 'trade', 'api_call', 'database')
            operation_data: Original operation data
            error: Exception that caused failure
            failure_reason: Categorized failure reason
            retry_count: Number of retries attempted
            metadata: Additional metadata

        Returns:
            Operation ID
        """
        operation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        entry = {
            'id': operation_id,
            'operation_type': operation_type,
            'operation_data': operation_data,
            'error': str(error),
            'error_type': type(error).__name__,
            'failure_reason': failure_reason.value,
            'retry_count': retry_count,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        # Save to file
        file_path = self._get_file_path(operation_id)
        try:
            with open(file_path, 'w') as f:
                json.dump(entry, f, indent=2)

            logger.error(
                f"Operation failed and added to DLQ: {operation_type} "
                f"(ID: {operation_id}, Reason: {failure_reason.value})"
            )

            # Send alert for critical failures
            if failure_reason in [FailureReason.AUTHENTICATION, FailureReason.SERVICE_UNAVAILABLE]:
                self._send_alert(entry)

            return operation_id

        except Exception as e:
            logger.error(f"Failed to add operation to DLQ: {e}")
            raise

    def get(self, operation_id: str) -> Optional[Dict]:
        """
        Get failed operation by ID.

        Args:
            operation_id: Operation ID

        Returns:
            Operation data or None if not found
        """
        file_path = self._get_file_path(operation_id)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to read DLQ entry {operation_id}: {e}")
            return None

    def list(
        self,
        operation_type: Optional[str] = None,
        failure_reason: Optional[FailureReason] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List failed operations.

        Args:
            operation_type: Filter by operation type
            failure_reason: Filter by failure reason
            limit: Maximum number of entries to return

        Returns:
            List of operations
        """
        operations = []

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    entry = json.load(f)

                # Apply filters
                if operation_type and entry['operation_type'] != operation_type:
                    continue

                if failure_reason and entry['failure_reason'] != failure_reason.value:
                    continue

                operations.append(entry)

                if len(operations) >= limit:
                    break

            except Exception as e:
                logger.error(f"Failed to read DLQ entry {file_path}: {e}")

        # Sort by timestamp (newest first)
        operations.sort(key=lambda x: x['timestamp'], reverse=True)

        return operations

    def retry(self, operation_id: str, retry_func: callable) -> bool:
        """
        Manually retry a failed operation.

        Args:
            operation_id: Operation ID
            retry_func: Function to retry the operation

        Returns:
            True if retry succeeded, False otherwise
        """
        entry = self.get(operation_id)
        if not entry:
            logger.error(f"Operation not found in DLQ: {operation_id}")
            return False

        try:
            logger.info(f"Retrying operation {operation_id}...")

            # Call retry function with original data
            retry_func(entry['operation_data'])

            # Success - remove from DLQ
            self.remove(operation_id)
            logger.info(f"Operation {operation_id} retried successfully")

            return True

        except Exception as e:
            logger.error(f"Retry failed for operation {operation_id}: {e}")

            # Update retry count
            entry['retry_count'] += 1
            entry['last_retry_time'] = datetime.utcnow().isoformat()
            entry['last_retry_error'] = str(e)

            file_path = self._get_file_path(operation_id)
            with open(file_path, 'w') as f:
                json.dump(entry, f, indent=2)

            return False

    def remove(self, operation_id: str) -> bool:
        """
        Remove operation from queue.

        Args:
            operation_id: Operation ID

        Returns:
            True if removed, False if not found
        """
        file_path = self._get_file_path(operation_id)
        try:
            file_path.unlink()
            logger.info(f"Removed operation from DLQ: {operation_id}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Failed to remove DLQ entry {operation_id}: {e}")
            return False

    def cleanup(self, max_age_days: int = 30) -> int:
        """
        Remove old entries from queue.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of entries removed
        """
        removed = 0
        cutoff_time = time.time() - (max_age_days * 86400)

        for file_path in self.storage_path.glob("*.json"):
            try:
                # Check file age
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed += 1

            except Exception as e:
                logger.error(f"Failed to cleanup DLQ entry {file_path}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old DLQ entries")

        return removed

    def get_stats(self) -> Dict:
        """
        Get statistics about failed operations.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total': 0,
            'by_type': {},
            'by_reason': {},
            'recent': []
        }

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    entry = json.load(f)

                stats['total'] += 1

                # Count by type
                op_type = entry['operation_type']
                stats['by_type'][op_type] = stats['by_type'].get(op_type, 0) + 1

                # Count by reason
                reason = entry['failure_reason']
                stats['by_reason'][reason] = stats['by_reason'].get(reason, 0) + 1

                # Track recent failures (last 24 hours)
                entry_time = datetime.fromisoformat(entry['timestamp'])
                age_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600

                if age_hours < 24:
                    stats['recent'].append({
                        'id': entry['id'],
                        'type': op_type,
                        'reason': reason,
                        'age_hours': round(age_hours, 1)
                    })

            except Exception as e:
                logger.error(f"Failed to process DLQ entry {file_path}: {e}")

        return stats

    def _get_file_path(self, operation_id: str) -> Path:
        """Get file path for operation ID."""
        return self.storage_path / f"{operation_id}.json"

    def _send_alert(self, entry: Dict):
        """Send alert for critical failures."""
        # In production, this would send email/Slack/PagerDuty alert
        logger.critical(
            f"CRITICAL: Operation failed - {entry['operation_type']} "
            f"(Reason: {entry['failure_reason']})"
        )


# ============================================================
# Helper Functions
# ============================================================

def dlq_on_failure(dlq: DeadLetterQueue, operation_type: str):
    """
    Decorator to automatically add failed operations to DLQ.

    Usage:
        @dlq_on_failure(dlq, "trade_execution")
        def execute_trade(data):
            # Trade logic
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add to DLQ
                operation_data = {
                    'args': args,
                    'kwargs': kwargs,
                    'function': func.__name__
                }

                dlq.add(
                    operation_type=operation_type,
                    operation_data=operation_data,
                    error=e,
                    failure_reason=classify_error(e)
                )
                raise
        return wrapper
    return decorator


def classify_error(error: Exception) -> FailureReason:
    """Classify error into failure reason category."""
    error_str = str(error).lower()

    if 'timeout' in error_str or 'timed out' in error_str:
        return FailureReason.TIMEOUT

    if 'rate limit' in error_str or '429' in error_str:
        return FailureReason.RATE_LIMIT

    if 'unauthorized' in error_str or '401' in error_str or 'forbidden' in error_str:
        return FailureReason.AUTHENTICATION

    if 'invalid' in error_str or 'bad request' in error_str or '400' in error_str:
        return FailureReason.INVALID_REQUEST

    if 'unavailable' in error_str or '503' in error_str or 'service' in error_str:
        return FailureReason.SERVICE_UNAVAILABLE

    return FailureReason.UNKNOWN


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("DEAD LETTER QUEUE TEST")
    print("=" * 70)

    # Create DLQ
    dlq = DeadLetterQueue("/tmp/trading_dlq")

    # Add some failed operations
    print("\n1. Adding failed operations:")

    op1 = dlq.add(
        operation_type="trade_execution",
        operation_data={
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001
        },
        error=TimeoutError("Request timed out after 10s"),
        failure_reason=FailureReason.TIMEOUT,
        retry_count=3
    )
    print(f"   Added: {op1}")

    op2 = dlq.add(
        operation_type="api_call",
        operation_data={"endpoint": "/account", "method": "GET"},
        error=Exception("Rate limit exceeded"),
        failure_reason=FailureReason.RATE_LIMIT,
        retry_count=5
    )
    print(f"   Added: {op2}")

    op3 = dlq.add(
        operation_type="database_write",
        operation_data={"table": "trades", "action": "INSERT"},
        error=ConnectionError("Database connection lost"),
        failure_reason=FailureReason.SERVICE_UNAVAILABLE,
        retry_count=2
    )
    print(f"   Added: {op3}")

    # List operations
    print("\n2. Listing failed operations:")
    operations = dlq.list()
    print(f"   Total: {len(operations)}")
    for op in operations:
        print(f"   - {op['operation_type']}: {op['failure_reason']} ({op['retry_count']} retries)")

    # Get statistics
    print("\n3. Statistics:")
    stats = dlq.get_stats()
    print(f"   Total operations: {stats['total']}")
    print(f"   By type: {stats['by_type']}")
    print(f"   By reason: {stats['by_reason']}")
    print(f"   Recent (24h): {len(stats['recent'])}")

    # Filter by type
    print("\n4. Filter by type (trade_execution):")
    trades = dlq.list(operation_type="trade_execution")
    print(f"   Found: {len(trades)} trade failures")

    # Filter by reason
    print("\n5. Filter by reason (RATE_LIMIT):")
    rate_limited = dlq.list(failure_reason=FailureReason.RATE_LIMIT)
    print(f"   Found: {len(rate_limited)} rate limit failures")

    # Manual retry example
    print("\n6. Manual retry:")
    def retry_trade(data):
        print(f"   Retrying trade: {data}")
        # In real implementation, this would execute the trade
        return True

    success = dlq.retry(op1, retry_trade)
    print(f"   Retry {'succeeded' if success else 'failed'}")

    # Cleanup old entries
    print("\n7. Cleanup (max age: 30 days):")
    removed = dlq.cleanup(max_age_days=30)
    print(f"   Removed {removed} old entries")

    print("\n✅ Dead letter queue test complete!")
