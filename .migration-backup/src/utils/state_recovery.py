"""
Agent State Recovery

Handles agent state persistence and recovery after crashes or restarts.
"""
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import shutil

logger = logging.getLogger(__name__)


class StateRecoveryManager:
    """
    Manages agent state persistence and recovery.

    Features:
    - Atomic state saves (write to temp, then rename)
    - State versioning
    - Backup management
    - Integrity verification
    - Automatic cleanup
    """

    def __init__(self, agent_id: str, storage_path: str = "/tmp/agent_state"):
        """
        Initialize state recovery manager.

        Args:
            agent_id: Unique agent identifier
            storage_path: Path to store state files
        """
        self.agent_id = agent_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.state_file = self.storage_path / f"{agent_id}.json"
        self.backup_dir = self.storage_path / f"{agent_id}_backups"
        self.backup_dir.mkdir(exist_ok=True)

        logger.info(f"State recovery manager initialized for agent: {agent_id}")

    def save_state(self, state: Dict, create_backup: bool = True) -> bool:
        """
        Save agent state atomically.

        Args:
            state: State dictionary to save
            create_backup: Whether to create a backup of previous state

        Returns:
            True if successful
        """
        try:
            # Add metadata
            state_with_meta = {
                'agent_id': self.agent_id,
                'timestamp': datetime.utcnow().isoformat(),
                'version': state.get('version', 1),
                'data': state
            }

            # Calculate checksum
            state_json = json.dumps(state_with_meta, sort_keys=True, indent=2)
            checksum = hashlib.sha256(state_json.encode()).hexdigest()
            state_with_meta['checksum'] = checksum

            # Backup existing state if requested
            if create_backup and self.state_file.exists():
                self._create_backup()

            # Atomic write: write to temp file, then rename
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_with_meta, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

            logger.info(f"Agent state saved: {self.agent_id} (v{state_with_meta['version']})")
            return True

        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            return False

    def load_state(self, verify_integrity: bool = True) -> Optional[Dict]:
        """
        Load agent state with integrity verification.

        Args:
            verify_integrity: Whether to verify checksum

        Returns:
            State dictionary or None if not found/corrupted
        """
        try:
            if not self.state_file.exists():
                logger.info(f"No saved state found for agent: {self.agent_id}")
                return None

            with open(self.state_file, 'r') as f:
                state_with_meta = json.load(f)

            # Verify integrity
            if verify_integrity:
                saved_checksum = state_with_meta.pop('checksum', None)
                state_json = json.dumps(state_with_meta, sort_keys=True, indent=2)
                calculated_checksum = hashlib.sha256(state_json.encode()).hexdigest()

                if saved_checksum != calculated_checksum:
                    logger.error(f"State integrity check failed for {self.agent_id}")
                    logger.info("Attempting to restore from backup...")
                    return self._restore_from_backup()

            state = state_with_meta['data']
            version = state_with_meta['version']
            timestamp = state_with_meta['timestamp']

            logger.info(
                f"Agent state loaded: {self.agent_id} "
                f"(v{version}, saved at {timestamp})"
            )

            return state

        except json.JSONDecodeError as e:
            logger.error(f"State file corrupted: {e}")
            return self._restore_from_backup()

        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return None

    def has_saved_state(self) -> bool:
        """Check if saved state exists."""
        return self.state_file.exists()

    def delete_state(self) -> bool:
        """Delete saved state and backups."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()

            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)

            logger.info(f"Agent state deleted: {self.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete agent state: {e}")
            return False

    def _create_backup(self) -> bool:
        """Create backup of current state."""
        try:
            if not self.state_file.exists():
                return False

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"backup_{timestamp}.json"

            shutil.copy2(self.state_file, backup_file)

            logger.debug(f"Created state backup: {backup_file.name}")

            # Keep only last 10 backups
            self._cleanup_old_backups(keep=10)

            return True

        except Exception as e:
            logger.error(f"Failed to create state backup: {e}")
            return False

    def _restore_from_backup(self) -> Optional[Dict]:
        """Restore state from most recent backup."""
        try:
            # Find most recent backup
            backups = sorted(self.backup_dir.glob("backup_*.json"), reverse=True)

            if not backups:
                logger.error("No backups available for restoration")
                return None

            backup_file = backups[0]
            logger.info(f"Restoring from backup: {backup_file.name}")

            with open(backup_file, 'r') as f:
                state_with_meta = json.load(f)

            # Restore to main state file
            shutil.copy2(backup_file, self.state_file)

            return state_with_meta['data']

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return None

    def _cleanup_old_backups(self, keep: int = 10):
        """Remove old backups, keeping only the most recent."""
        try:
            backups = sorted(self.backup_dir.glob("backup_*.json"), reverse=True)

            for backup in backups[keep:]:
                backup.unlink()
                logger.debug(f"Removed old backup: {backup.name}")

        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

    def get_state_info(self) -> Optional[Dict]:
        """Get metadata about saved state without loading it."""
        try:
            if not self.state_file.exists():
                return None

            with open(self.state_file, 'r') as f:
                state_with_meta = json.load(f)

            return {
                'agent_id': state_with_meta['agent_id'],
                'timestamp': state_with_meta['timestamp'],
                'version': state_with_meta['version'],
                'file_size': self.state_file.stat().st_size,
                'backup_count': len(list(self.backup_dir.glob("backup_*.json")))
            }

        except Exception as e:
            logger.error(f"Failed to get state info: {e}")
            return None


class RecoverableAgentMixin:
    """
    Mixin to add state recovery capabilities to trading agents.

    Usage:
        class MyAgent(RecoverableAgentMixin):
            def __init__(self, agent_id):
                self.state_manager = StateRecoveryManager(agent_id)
                self.recover_state()
    """

    def save_checkpoint(self):
        """Save current agent state as checkpoint."""
        state = self._get_recoverable_state()
        return self.state_manager.save_state(state)

    def recover_state(self) -> bool:
        """
        Recover agent state from checkpoint.

        Returns:
            True if state was recovered, False otherwise
        """
        state = self.state_manager.load_state()

        if not state:
            logger.info(f"No state to recover for {self.agent_id}")
            return False

        try:
            self._restore_from_state(state)
            logger.info(f"Agent state recovered: {self.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore agent state: {e}")
            return False

    def _get_recoverable_state(self) -> Dict:
        """
        Get current state that should be saved for recovery.

        Override this method to customize what state is saved.
        """
        return {
            'version': 1,
            'agent_id': getattr(self, 'agent_id', 'unknown'),
            'status': getattr(self, 'status', 'unknown'),
            'portfolio_value': getattr(self, 'portfolio_value', 0.0),
            'positions': getattr(self, 'positions', {}),
            'open_orders': getattr(self, 'open_orders', []),
            'metrics': getattr(self, 'metrics', {})
        }

    def _restore_from_state(self, state: Dict):
        """
        Restore agent from saved state.

        Override this method to customize how state is restored.
        """
        if 'portfolio_value' in state:
            self.portfolio_value = state['portfolio_value']

        if 'positions' in state:
            self.positions = state['positions']

        if 'open_orders' in state:
            self.open_orders = state['open_orders']

        if 'metrics' in state:
            self.metrics = state['metrics']


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("STATE RECOVERY TEST")
    print("=" * 70)

    # Create state manager
    agent_id = "test-agent-001"
    manager = StateRecoveryManager(agent_id, "/tmp/test_state")

    # Test 1: Save state
    print("\n1. Saving agent state:")
    state = {
        'version': 1,
        'portfolio_value': 10000.0,
        'positions': {
            'BTCUSDT': {'quantity': 0.1, 'avg_price': 45000.0},
            'ETHUSDT': {'quantity': 2.0, 'avg_price': 2500.0}
        },
        'open_orders': [
            {'order_id': '12345', 'symbol': 'BTCUSDT', 'side': 'BUY'},
            {'order_id': '12346', 'symbol': 'ETHUSDT', 'side': 'SELL'}
        ],
        'metrics': {
            'total_trades': 42,
            'win_rate': 0.65,
            'sharpe_ratio': 1.8
        }
    }

    success = manager.save_state(state)
    print(f"   State saved: {success}")

    # Test 2: Get state info
    print("\n2. State information:")
    info = manager.get_state_info()
    if info:
        print(f"   Agent ID: {info['agent_id']}")
        print(f"   Version: {info['version']}")
        print(f"   Timestamp: {info['timestamp']}")
        print(f"   File size: {info['file_size']} bytes")
        print(f"   Backups: {info['backup_count']}")

    # Test 3: Load state
    print("\n3. Loading agent state:")
    loaded_state = manager.load_state()
    if loaded_state:
        print(f"   Portfolio value: ${loaded_state['portfolio_value']:,.2f}")
        print(f"   Positions: {len(loaded_state['positions'])}")
        print(f"   Open orders: {len(loaded_state['open_orders'])}")
        print(f"   Total trades: {loaded_state['metrics']['total_trades']}")

    # Test 4: Update and save again (creates backup)
    print("\n4. Updating state (creates backup):")
    state['version'] = 2
    state['portfolio_value'] = 12500.0
    state['metrics']['total_trades'] = 50

    manager.save_state(state)
    info = manager.get_state_info()
    print(f"   New version: {info['version']}")
    print(f"   Backups created: {info['backup_count']}")

    # Test 5: Test integrity verification
    print("\n5. Testing integrity verification:")
    loaded_state = manager.load_state(verify_integrity=True)
    print(f"   Integrity check: {'passed' if loaded_state else 'failed'}")

    # Test 6: Recoverable agent example
    print("\n6. Recoverable agent example:")

    class TestAgent(RecoverableAgentMixin):
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.state_manager = StateRecoveryManager(agent_id, "/tmp/test_state")
            self.portfolio_value = 5000.0
            self.positions = {}
            self.open_orders = []
            self.metrics = {}

    agent = TestAgent("recoverable-agent-001")

    # Set some state
    agent.portfolio_value = 15000.0
    agent.positions = {'BTCUSDT': {'quantity': 0.2, 'avg_price': 46000.0}}

    # Save checkpoint
    print("   Saving checkpoint...")
    agent.save_checkpoint()

    # Simulate crash and recovery
    print("   Simulating crash...")
    agent.portfolio_value = 0.0
    agent.positions = {}

    print("   Recovering state...")
    recovered = agent.recover_state()
    print(f"   Recovery {'successful' if recovered else 'failed'}")
    print(f"   Portfolio value: ${agent.portfolio_value:,.2f}")
    print(f"   Positions: {len(agent.positions)}")

    # Cleanup
    print("\n7. Cleanup:")
    manager.delete_state()
    print("   State files deleted")

    print("\n✅ State recovery test complete!")
