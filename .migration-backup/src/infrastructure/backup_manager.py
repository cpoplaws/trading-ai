"""
Database Backup Manager - Automated PostgreSQL backups with S3 storage

Features:
- Scheduled daily backups to S3
- On-demand manual backups
- Compression with pg_dump
- Backup retention policy
- Restore from backup
- Backup verification
- Backup cleanup
"""

import os
import logging
import subprocess
import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from contextlib import contextmanager
import asyncio
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Configuration for database backups."""
    # Schedule settings
    enabled: bool = True
    time: str = "02:00"  # HH:MM format, 24h
    timezone: str = "UTC"

    # Retention settings
    daily_backups: int = 7  # Keep daily backups for 7 days
    weekly_backups: int = 4  # Keep weekly backups for 4 weeks
    monthly_backups: int = 6  # Keep monthly backups for 6 months

    # Storage settings
    s3_bucket: str = os.getenv('S3_BUCKET', 'trading-ai-backups')
    s3_prefix: str = "backups/"
    local_backup_dir: str = "/tmp/trading-ai-backups"
    compress_backups: bool = True  # Compress backup files

    # Notification settings
    on_backup_success: bool = False  # Slack webhook on backup success
    on_backup_failure: bool = True  # Slack webhook on backup failure
    slack_webhook_url: str = ""
    email_recipients: List[str] = None

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class BackupInfo:
    """Information about a database backup."""
    id: str
    type: str  # 'scheduled', 'manual', 'on_demand'
    timestamp: datetime.datetime
    size_bytes: int
    filename: str
    duration_seconds: int
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    error_message: Optional[str] = None
    tables_included: List[str] = None
    pg_dump_path: Optional[str] = None
    s3_path: Optional[str] = None  # S3 path if applicable
    checksum: str = ""

    def __post_init__(self):
        if self.tables_included is None:
            self.tables_included = []


class DatabaseBackupManager:
    """
    Manages automated database backups with PostgreSQL pg_dump.

    Supports:
    - Scheduled daily backups to S3
    - Manual backup on demand
    - Compression with pg_dump
    - Restore from backup
    - Backup verification and cleanup

    Backup strategy:
    1. Use pg_dump with compression
    2. Store compressed backups to S3
    3. Keep backup retention per policy
    4. Verify backup integrity
    """

    def __init__(self, db_host: str, db_name: str, db_user: str, db_password: str, config: BackupConfig = None):
        """
        Initialize backup manager.

        Args:
            db_host: Database host
            db_name: Database name
            db_user: Database user
            db_password: Database password
            config: Backup configuration (optional)
        """
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.config = config or BackupConfig()
        self._backup_dir = self.config.local_backup_dir
        self._s3_client = None  # boto3 client for S3
        self._backups: List[BackupInfo] = []

        # Ensure backup directory exists
        os.makedirs(self._backup_dir, exist_ok=True)

        logger.info(f"Database Backup Manager initialized for {self.db_name}")

    def _get_s3_client(self) -> Optional[object]:
        """Get or create S3 client."""
        if self._s3_client is None:
            try:
                import boto3
                session = boto3.Session(profile_name='trading-ai-backups')
                self._s3_client = session.client('s3')
                logger.info(f"S3 client initialized")
                return self._s3_client
            except ImportError:
                logger.warning("boto3 not installed. S3 backups will use local storage.")
                return None
            except Exception as e:
                logger.error(f"S3 client error: {e}")
                return None
        return self._s3_client

    async def create_daily_backup(self) -> BackupInfo:
        """
        Create a daily database backup.

        Returns:
            BackupInfo object with backup details
        """
        if not self.config.enabled:
            raise ValueError("Database backups are disabled")

        timestamp = datetime.datetime.now()
        filename = f"daily_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.sql"
        filepath = os.path.join(self._backup_dir, filename)

        try:
            # Use pg_dump with compression
            dump_result = await self._run_backup_command(filepath)

            backup_info = BackupInfo(
                id=filename,
                type='scheduled',
                timestamp=timestamp,
                size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                filename=filename,
                duration_seconds=dump_result.get('duration_seconds', 0),
                start_time=timestamp,
                end_time=datetime.datetime.now(),
                status='completed',
                tables_included=dump_result.get('tables_included', []),
                pg_dump_path=filepath,
                s3_path=None,
                checksum=dump_result.get('checksum', '')
            )

            # Update tracking
            self._backups.append(backup_info)

            logger.info(f"Daily backup created: {filename}")

            # Upload to S3 if configured
            if self.config.s3_bucket:
                s3_success = await self._upload_to_s3(filepath)
                if s3_success:
                    s3_path = f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}{filename}"
                    backup_info.s3_path = s3_path

            return backup_info

        except Exception as e:
            logger.error(f"Daily backup failed: {e}")

            # Create failed backup info
            return BackupInfo(
                id=filename,
                type='scheduled',
                timestamp=timestamp,
                size_bytes=0,
                filename=filename,
                duration_seconds=0,
                start_time=timestamp,
                end_time=datetime.datetime.now(),
                status='failed',
                error_message=str(e),
                tables_included=[],
                pg_dump_path=None,
                s3_path=None,
                checksum='',
            )

    async def create_manual_backup(self, tables: Optional[List[str]] = None) -> BackupInfo:
        """
        Create an on-demand manual backup.

        Args:
            tables: List of table names to backup (default: all tables)

        Returns:
            BackupInfo object with backup details
        """
        timestamp = datetime.datetime.now()

        # Generate filename
        filename = f"manual_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.sql"
        filepath = os.path.join(self._backup_dir, filename)

        try:
            # Run backup
            dump_result = await self._run_backup_command(filepath, tables=tables)

            backup_info = BackupInfo(
                id=filename,
                type='manual',
                timestamp=timestamp,
                size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                filename=filename,
                duration_seconds=dump_result.get('duration_seconds', 0),
                start_time=timestamp,
                end_time=datetime.datetime.now(),
                status='completed',
                tables_included=tables or ['all'],
                pg_dump_path=filepath,
                s3_path=None,
                checksum=dump_result.get('checksum', '')
            )

            # Update tracking
            self._backups.append(backup_info)

            logger.info(f"Manual backup created: {filename}")

            # Upload to S3 if configured
            if self.config.s3_bucket:
                s3_success = await self._upload_to_s3(filepath)
                if s3_success:
                    s3_path = f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}{filename}"
                    backup_info.s3_path = s3_path

            return backup_info

        except Exception as e:
            logger.error(f"Manual backup failed: {e}")

            # Create failed backup info
            return BackupInfo(
                id=filename,
                type='manual',
                timestamp=timestamp,
                size_bytes=0,
                filename=filename,
                duration_seconds=0,
                start_time=timestamp,
                end_time=datetime.datetime.now(),
                status='failed',
                error_message=str(e),
                tables_included=[],
                pg_dump_path=None,
                s3_path=None,
                checksum='',
            )

    async def _run_backup_command(
        self,
        filepath: str,
        tables: Optional[List[str]] = None
    ) -> Dict:
        """
        Execute pg_dump backup command.

        Returns:
            Dict with backup results
        """
        # Build command
        pg_dump_cmd = [
            'pg_dump',
            f'--host={self.db_host}',
            f'--port=5432',
            f'--username={self.db_user}',
            f'--dbname={self.db_name}',
            '--format=plain',
            '--no-owner',
            '--no-acl',
            f'--file={filepath}',
        ]

        # Set password in environment
        env = os.environ.copy()
        env["PGPASSWORD"] = self.db_password

        logger.debug(f"Running pg_dump for {self.db_name}")

        # Execute command
        start_time = datetime.datetime.now()
        process = subprocess.Popen(
            pg_dump_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        end_time = datetime.datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        if process.returncode != 0:
            logger.error(f"pg_dump failed: {stderr.decode()}")
            raise RuntimeError(f"Backup failed: {stderr.decode()}")

        # Compress if configured
        if self.config.compress_backups:
            import gzip
            with open(filepath, 'rb') as f_in:
                with gzip.open(filepath + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(filepath)
            filepath = filepath + '.gz'

        return {
            'duration_seconds': int(duration_seconds),
            'tables_included': tables or ['all'],
            'size_bytes': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            'checksum': self._calculate_checksum(filepath) if os.path.exists(filepath) else '',
            'returncode': process.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of file."""
        import hashlib
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    async def _upload_to_s3(self, filepath: str) -> bool:
        """
        Upload backup to S3 storage.

        Args:
            filepath: Path to backup file

        Returns:
            True if successful, False otherwise
        """
        if not self.config.s3_bucket:
            logger.warning("S3 bucket not configured, backup will be local only")
            return False

        # Get S3 client
        s3_client = self._get_s3_client()
        if not s3_client:
            logger.error("S3 client not available, cannot upload to S3")
            return False

        try:
            import boto3

            # Upload file
            with open(filepath, 'rb') as f:
                file_data = f.read()

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}{os.path.basename(filepath)}"
            s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=file_data,
                ContentType='application/gzip',
                Metadata={'filename': os.path.basename(filepath)}
            )

            logger.info(f"Uploaded to S3: {s3_key}")
            return True

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False

    def get_backup_list(self, limit: int = 10) -> List[BackupInfo]:
        """
        Get list of backups.

        Args:
            limit: Maximum number of backups to return

        Returns:
            List of backup information
        """
        return self._backups[-limit:] if self._backups else []

    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """
        Get detailed information about a specific backup.

        Args:
            backup_id: Backup ID (filename without extension)

        Returns:
            BackupInfo or None if found
        """
        for backup in self._backups:
            if backup.id == backup_id:
                return backup
        return None

    async def restore_backup(self, backup_id: str) -> bool:
        """
        Restore database from backup.

        Args:
            backup_id: Backup ID (filename)

        Returns:
            True if successful, False otherwise
        """
        # Find backup
        backup = self.get_backup_info(backup_id)
        if not backup:
            logger.error(f"Backup not found: {backup_id}")
            return False

        # Find backup file
        backup_file = backup.pg_dump_path
        if not backup_file or not os.path.exists(backup_file):
            logger.error(f"Backup file not found: {backup_file}")

            # Try to download from S3
            if backup.s3_path and self._s3_client:
                return await self._restore_from_s3(backup)

            return False

        # Restore from file
        return await self._restore_from_file(backup_file)

    async def _restore_from_file(self, backup_file: str) -> bool:
        """
        Restore database from local backup file.

        Args:
            backup_file: Path to backup file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build restore command
            restore_cmd = [
                'psql',
                f'--host={self.db_host}',
                f'--port=5432',
                f'--username={self.db_user}',
                f'--dbname={self.db_name}',
            ]

            # Set password in environment
            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_password

            # Handle gzip compression
            import gzip
            if backup_file.endswith('.gz'):
                with gzip.open(backup_file, 'rb') as f_in:
                    process = subprocess.Popen(
                        restore_cmd,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = process.communicate(input=f_in.read())
            else:
                with open(backup_file, 'rb') as f_in:
                    process = subprocess.Popen(
                        restore_cmd,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    stdout, stderr = process.communicate(input=f_in.read())

            if process.returncode != 0:
                logger.error(f"Restore failed: {stderr.decode()}")
                return False

            logger.info(f"Restore completed from {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    async def _restore_from_s3(self, backup: BackupInfo) -> bool:
        """
        Restore database from S3 backup.

        Args:
            backup: BackupInfo object

        Returns:
            True if successful, False otherwise
        """
        try:
            # Download from S3
            local_file = os.path.join(self._backup_dir, backup.filename)

            s3_key = backup.s3_path.split('/')[-1]
            self._s3_client.download_file(
                Bucket=self.config.s3_bucket,
                Key=f"{self.config.s3_prefix}{backup.filename}",
                Filename=local_file
            )

            logger.info(f"Downloaded from S3 to {local_file}")

            # Restore from downloaded file
            return await self._restore_from_file(local_file)

        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    async def cleanup_old_backups(self, older_than_days: int = 30) -> Dict[str, any]:
        """
        Delete old backups beyond retention policy.

        Args:
            older_than_days: Delete backups older than this many days

        Returns:
            Summary of cleanup operation
        """
        if not self.config.enabled:
            return {"status": "disabled", "deleted_count": 0, "error": None}

        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)

        try:
            deleted_count = 0
            error_messages = []

            for backup in self._backups[:]:  # Copy list to iterate
                if backup.timestamp < cutoff_date:
                    # Delete local file
                    backup_file = backup.pg_dump_path
                    if backup_file and os.path.exists(backup_file):
                        os.remove(backup_file)
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup.filename}")

                    # Delete from S3 if configured
                    if backup.s3_path and self._s3_client:
                        try:
                            s3_key = backup.s3_path.split('/')[-1]
                            self._s3_client.delete_object(
                                Bucket=self.config.s3_bucket,
                                Key=f"{self.config.s3_prefix}{backup.filename}"
                            )
                            logger.info(f"Deleted from S3: {backup.filename}")
                        except Exception as e:
                            error_messages.append(str(e))
                            logger.warning(f"Failed to delete from S3: {e}")

                    # Remove from tracking
                    self._backups.remove(backup)

            return {
                "status": "completed",
                "deleted_count": deleted_count,
                "error_messages": error_messages
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                "status": "failed",
                "deleted_count": deleted_count,
                "error": str(e)
            }

    def verify_backup(self, backup_id: str) -> Dict[str, str]:
        """
        Verify backup integrity using checksum.

        Args:
            backup_id: Backup ID (filename)

        Returns:
            Verification result dictionary
        """
        backup = self.get_backup_info(backup_id)
        if not backup:
            return {
                "backup_id": backup_id,
                "status": "error",
                "message": "Backup not found"
            }

        # Check local file exists
        local_file = backup.pg_dump_path
        if not local_file or not os.path.exists(local_file):
            return {
                "backup_id": backup_id,
                "status": "error",
                "message": "Local file not found"
            }

        # Verify checksum
        if backup.checksum:
            try:
                local_checksum = self._calculate_checksum(local_file)
                if local_checksum != backup.checksum:
                    return {
                        "backup_id": backup_id,
                        "status": "error",
                        "message": f"Checksum mismatch: local={local_checksum[:16]} != backup={backup.checksum[:16]}"
                    }
            except Exception as e:
                return {
                    "backup_id": backup_id,
                    "status": "error",
                    "message": str(e)
                }

        # Check S3 if configured
        if backup.s3_path and self._s3_client:
            try:
                s3_key = backup.s3_path.split('/')[-1]
                self._s3_client.head_object(
                    Bucket=self.config.s3_bucket,
                    Key=f"{self.config.s3_prefix}{backup.filename}"
                )
            except Exception as e:
                return {
                    "backup_id": backup_id,
                    "status": "warning",
                    "message": f"S3 object check failed: {e}"
                }

        return {
            "backup_id": backup_id,
            "status": "verified",
            "message": "Backup verified successfully"
        }


def main():
    """
    Entry point for standalone backup manager operations.

    Usage:
        python -m src.infrastructure.backup_manager daily
        python -m src.infrastructure.backup_manager manual
        python -m src.infrastructure.backup_manager restore <backup_id>
        python -m src.infrastructure.backup_manager list
        python -m src.infrastructure.backup_manager cleanup
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.infrastructure.backup_manager <command>")
        print("Commands: daily, manual, restore, list, cleanup, verify")
        sys.exit(1)

    command = sys.argv[1]

    # Initialize backup manager
    config = BackupConfig(
        db_host=os.getenv('DB_HOST', 'localhost'),
        db_name=os.getenv('DB_NAME', 'trading_ai'),
        db_user=os.getenv('DB_USER', 'postgres'),
        db_password=os.getenv('DB_PASSWORD', ''),
        s3_bucket=os.getenv('S3_BUCKET'),
    )

    manager = DatabaseBackupManager(
        db_host=config.db_host,
        db_name=config.db_name,
        db_user=config.db_user,
        db_password=config.db_password,
        config=config
    )

    async def run_command():
        if command == "daily":
            backup = await manager.create_daily_backup()
            print(f"Backup created: {backup.id}")

        elif command == "manual":
            tables = sys.argv[2:] if len(sys.argv) > 2 else None
            backup = await manager.create_manual_backup(tables)
            print(f"Backup created: {backup.id}")

        elif command == "restore":
            if len(sys.argv) < 3:
                print("Usage: python -m src.infrastructure.backup_manager restore <backup_id>")
                sys.exit(1)
            backup_id = sys.argv[2]
            success = await manager.restore_backup(backup_id)
            print(f"Restore {'successful' if success else 'failed'}")

        elif command == "list":
            backups = manager.get_backup_list()
            print(f"Backups ({len(backups)}):")
            for backup in backups:
                print(f"  {backup.id} - {backup.timestamp} - {backup.status}")

        elif command == "cleanup":
            result = await manager.cleanup_old_backups()
            print(f"Cleanup: {result['status']}, deleted: {result['deleted_count']}")

        elif command == "verify":
            if len(sys.argv) < 3:
                print("Usage: python -m src.infrastructure.backup_manager verify <backup_id>")
                sys.exit(1)
            backup_id = sys.argv[2]
            result = manager.verify_backup(backup_id)
            print(f"Verify: {result['status']} - {result['message']}")

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    asyncio.run(run_command())


if __name__ == "__main__":
    main()
