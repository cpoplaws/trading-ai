# Trading AI - Disaster Recovery Plan

## Overview

This document outlines comprehensive procedures for recovering from various failure scenarios in the Trading AI production system. This plan is designed to minimize downtime, protect trading assets, and ensure data integrity.

**Document Version:** 1.0
**Last Updated:** 2026-03-08
**Owner:** DevOps Team
**Review Frequency:** Quarterly

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Recovery Objectives](#recovery-objectives)
3. [Failure Scenarios](#failure-scenarios)
4. [Backup Strategy](#backup-strategy)
5. [Recovery Procedures](#recovery-procedures)
6. [Communication Plan](#communication-plan)
7. [Testing and Validation](#testing-and-validation)
8. [Contacts](#contacts)

---

## Executive Summary

The Trading AI system is designed with redundancy and fault tolerance at its core. However, this DRP provides documented procedures for handling catastrophic failures that cannot be automatically recovered.

**Key Principles:**
- **Safety First:** Always halt trading if system integrity is uncertain
- **Data First:** Prioritize data integrity over quick recovery
- **Communication First:** Inform stakeholders of issues immediately
- **Document Everything:** Log all recovery actions for post-mortem analysis

---

## Recovery Objectives

### Recovery Time Objectives (RTO)

| Component | RTO | Notes |
|-----------|-----|-------|
| Database (PostgreSQL) | 1 hour | Critical for trading operations |
| Redis Cache | 5 minutes | Can operate in degraded mode |
| Trading Bot Core | 10 minutes | Must halt if unsafe |
| API Services | 30 minutes | Affects monitoring |
| Dashboard | 30 minutes | Non-critical |

### Recovery Point Objectives (RPO)

| Component | RPO | Notes |
|-----------|-----|-------|
| Database | 15 minutes | Daily automated backups |
| Redis | 1 minute | Ephemeral data, can be rebuilt |
| Trading Positions | Real-time | Exchange API provides source of truth |
| Configuration | Instant | Version controlled in git |

### Maximum Tolerable Downtime (MTD)

**System-wide MTD:** 4 hours
**Trading operations MTD:** 2 hours

If downtime exceeds MTD, escalate to management team for business continuity decisions.

---

## Failure Scenarios

### Scenario 1: Database Failure

**Severity:** CRITICAL
**Probability:** Low
**Impact:** Total system outage

#### Symptoms
- API returns 500 errors
- Dashboard shows connection errors
- Pre-launch validator fails connectivity checks
- All trading agents error with database exceptions

#### Recovery Steps

**Step 1: Diagnose (0-15 minutes)**
```bash
# Check PostgreSQL status
docker-compose ps postgres
docker logs postgres --tail 100

# Check database connectivity
python -c "from src.database.database_manager import DatabaseManager; db = DatabaseManager(); print('Connected' if db.test_connection() else 'Failed')"

# Check disk space
df -h
```

**Step 2: Attempt Automated Recovery (15-30 minutes)**
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check if recovery successful
docker-compose logs postgres --tail 50
```

**Step 3: Restore from Backup if needed (30-60 minutes)**
```bash
# Download latest backup from S3
aws s3 cp s3://trading-ai-backups/backups/latest_backup.sql.gz /tmp/

# Restore backup
python -m src.infrastructure.backup_manager restore latest_backup.sql.gz

# Verify data integrity
python -m src.deployment.prelaunch_validator --check database_integrity
```

**Step 4: Resume Trading (60-90 minutes)**
```bash
# Restart trading agents
python start.py --mode live

# Monitor for issues
tail -f logs/trading.log
```

#### Rollback Procedures
If restore fails:
1. Use backup from 24 hours prior
2. Manually reconstruct positions from exchange APIs
3. Re-run pre-launch validator
4. Conduct manual review before resuming trading

#### Prevention
- Daily automated backups
- Database replication setup
- Regular integrity checks
- Monitor disk usage alerts

---

### Scenario 2: Redis Cache Failure

**Severity:** MEDIUM
**Probability:** Medium
**Impact:** Degraded performance, lost cache data

#### Symptoms
- Slow API responses
- Cache errors in logs
- Market data stale or missing

#### Recovery Steps

**Step 1: Diagnose (0-5 minutes)**
```bash
# Check Redis status
docker-compose ps redis
docker logs redis --tail 50

# Test connectivity
redis-cli -h localhost -p 6379 ping
```

**Step 2: Restart Redis (5-10 minutes)**
```bash
# Restart Redis
docker-compose restart redis

# Clear corrupted cache if needed
redis-cli FLUSHALL
```

**Step 3: Verify (10-15 minutes)**
```bash
# Test cache operations
python -c "from src.infrastructure.market_data_cache import MarketDataCache; c = MarketDataCache(); c.set('test', 'value', 10); print(c.get('test'))"
```

#### Rollback Procedures
- Not applicable - cache is ephemeral and can be rebuilt

#### Prevention
- Redis persistence (AOF + RDB)
- Memory usage monitoring
- Regular cache warmup after restarts

---

### Scenario 3: Exchange API Outage

**Severity:** HIGH
**Probability:** Medium
**Impact:** Trading execution failures on affected exchange

#### Symptoms
- Trading execution failures
- Order placement errors
- Circuit breaker triggers
- "Exchange unavailable" alerts

#### Recovery Steps

**Step 1: Identify Affected Exchange (0-5 minutes)**
```bash
# Check exchange status pages
# Binance: https://status.binance.com/
# Coinbase: https://status.coinbase.com/

# Check logs for specific errors
grep "exchange" logs/trading.log | tail -50
```

**Step 2: Pause Trading on Affected Exchange (5-10 minutes)**
```python
# Manually pause via dashboard or API
curl -X POST http://localhost:8000/api/agents/pause \
  -H "Content-Type: application/json" \
  -d '{"exchange": "binance"}'
```

**Step 3: Switch to Alternative Routes (10-20 minutes)**
- If DEX available: Use Uniswap/Jupiter routes
- If multiple CEX: Shift volume to other exchanges
- Update strategy parameters to avoid affected exchange

**Step 4: Monitor for Exchange Status Updates (20-120 minutes)**
- Subscribe to exchange status feeds
- Monitor for restoration announcements
- Do not resume trading until confirmed operational

#### Rollback Procedures
- Cancel pending orders on affected exchange
- Close positions at best available price if necessary
- Document any realized losses

#### Prevention
- Circuit breaker logic (already implemented)
- Multi-exchange diversification
- Status page monitoring alerts
- Manual override capabilities

---

### Scenario 4: Network Outage

**Severity:** MEDIUM-HIGH
**Probability:** Low
**Impact:** No external API access, trading halted

#### Symptoms
- All API calls timeout
- "Connection refused" or "Network unreachable" errors
- Grafana shows loss of external connectivity

#### Recovery Steps

**Step 1: Verify Network Status (0-5 minutes)**
```bash
# Check internet connectivity
ping -c 3 8.8.8.8

# Check DNS resolution
nslookup api.binance.com

# Check local network
ifconfig | grep status
```

**Step 2: Troubleshoot Network (5-15 minutes)**
- Check firewall rules
- Verify VPN/proxy configuration
- Check with ISP if external issue
- Restart network services if needed

**Step 3: Halt Trading Operations (5-10 minutes)**
```bash
# Stop all trading agents
python -m src.autonomous_agent.trading_agent --stop

# Verify agents stopped
python start.py --status
```

**Step 4: Resume When Network Restored (Variable)**
- Run pre-launch validator
- Check exchange connectivity
- Resume trading gradually

#### Rollback Procedures
- Not applicable - no action possible during outage

#### Prevention
- Multi-cloud deployment
- Network monitoring alerts
- Redundant internet connections (production)

---

### Scenario 5: API Key Compromise

**Severity:** CRITICAL
**Probability:** Low
**Impact:** Unauthorized trading, potential losses

#### Symptoms
- Unauthorized transactions detected
- API errors due to invalid credentials
- Security audit alerts triggered
- Unexpected position changes

#### Recovery Steps

**Step 1: Immediate Response (0-5 minutes)**
```bash
# HALT ALL TRADING OPERATIONS
python -m src.autonomous_agent.trading_agent --emergency-stop

# Disable compromised API keys via exchange dashboards
# Binance: https://www.binance.com/en/my/settings/api-management
# Coinbase: https://www.coinbase.com/settings/api
```

**Step 2: Rotate All API Keys (5-30 minutes)**
```bash
# Generate new keys via exchange
# Update in secrets manager
python -m src.infrastructure.secrets_manager rotate --service binance
python -m src.infrastructure.secrets_manager rotate --service coinbase

# Remove old keys from exchanges
```

**Step 3: Conduct Security Audit (30-120 minutes)**
```python
# Run security audit
python -m src.security.security_audit

# Review recent trades
python start.py --audit --days 7

# Check for unauthorized positions
curl http://localhost:8000/api/portfolio/positions
```

**Step 4: Restore from Pre-Compromise Backup (if needed)**
```bash
# Identify compromise time
# Restore backup from before compromise
python -m src.infrastructure.backup_manager restore <pre_compromise_backup_id>
```

**Step 5: Gradual Resumption (120-180 minutes)**
- Start in paper trading mode
- Monitor for unauthorized activity
- Gradually increase to live trading
- Continue enhanced monitoring for 48 hours

#### Rollback Procedures
- If new keys also compromised: Complete system shutdown, contact exchange support
- If data corruption suspected: Restore from verified clean backup

#### Prevention
- Use AWS Secrets Manager with rotation
- IP whitelisting for API access
- Multi-factor authentication
- Regular security audits
- Monitor for unusual activity patterns

---

### Scenario 6: Server Failure

**Severity:** CRITICAL
**Probability:** Low
**Impact:** Total system outage

#### Symptoms
- No server response
- All services down
- Infrastructure alerts triggered

#### Recovery Steps

**Step 1: Identify Failure Type (0-10 minutes)**
- Check if AWS/DigitalOcean/Azure console shows server running
- Check if issue is specific to one service or entire server
- Verify if backup/failover server available

**Step 2: Attempt Service Restart (10-20 minutes)**
```bash
# If server running but services down
docker-compose restart

# If individual services failed
docker-compose restart <service_name>
```

**Step 3: Failover to Backup Server (20-60 minutes)**
```bash
# If primary server unrecoverable
# Point DNS to backup server
# Restore latest backup to backup server
python -m src.infrastructure.backup_manager restore latest

# Verify all services running
docker-compose ps
```

#### Rollback Procedures
- If failover server also fails: Escalate to management, declare major incident

#### Prevention
- Multi-AZ deployment
- Auto-scaling groups
- Health checks and auto-recovery
- Regular failover drills

---

## Backup Strategy

### Backup Schedule

| Backup Type | Frequency | Retention | Location |
|-------------|-----------|-----------|----------|
| Database (Full) | Daily @ 2 AM UTC | 7 days | S3 + Local |
| Database (Weekly) | Weekly @ 2 AM Sunday | 4 weeks | S3 |
| Database (Monthly) | 1st of month | 12 months | S3 |
| Configuration | On git push | Indefinite | GitHub |
| Logs | Daily | 30 days | S3 |
| Application Code | On git push | Indefinite | GitHub |

### Backup Storage

**Primary Storage:**
- AWS S3 (us-east-1) - `trading-ai-backups` bucket
- Versioning enabled
- Cross-region replication to eu-west-1

**Secondary Storage:**
- Local backups on production server (last 7 days)
- Off-site backup tape (monthly)

**Access Control:**
- S3 bucket encrypted with AES-256
- IAM role with least privilege
- MFA required for backup access
- Audit logging enabled

### Backup Verification

**Daily:**
- Automated backup success/failure notifications
- Checksum verification on all backups

**Weekly:**
- Restore test to staging environment
- Data integrity checks

**Monthly:**
- Full disaster recovery drill
- RTO/RPO validation

---

## Recovery Procedures

### Pre-Recovery Checklist

Before initiating recovery:
- [ ] Incident identified and documented
- [ ] All stakeholders notified
- [ ] Trading operations halted if unsafe
- [ ] Root cause identified
- [ ] Recovery team assembled
- [ ] Recovery plan selected
- [ ] Rollback plan prepared

### Recovery Execution

1. **Execute Recovery Steps** - Follow scenario-specific procedures
2. **Monitor Progress** - Track against RTO targets
3. **Verify Integrity** - Run validation checks
4. **Resume Operations** - Gradual return to normal
5. **Monitor** - Enhanced monitoring for 24-48 hours

### Post-Recovery

1. **Document Incident**
   - Root cause analysis
   - Timeline of events
   - Actions taken
   - Lessons learned

2. **Update DRP**
   - Document gaps discovered
   - Update procedures based on learnings
   - Adjust RTO/RPO if needed

3. **Conduct Post-Mortem**
   - Team meeting within 7 days
   - Action items assigned
   - Tracking established

---

## Communication Plan

### Communication Channels

| Audience | Channel | Frequency | Trigger |
|----------|---------|-----------|----------|
| Internal Team | Slack #incidents | Real-time | Incident detected |
| Management | Email + PagerDuty | Immediately | Severity > MEDIUM |
| Customers | Status Page + Email | Within 1 hour | Service impact |
| Public | Twitter/Discord | As needed | Major outage |

### Communication Templates

**Initial Incident Notification:**
```
INCIDENT ALERT
Severity: [LOW|MEDIUM|HIGH|CRITICAL]
Time: [timestamp]
Description: [brief description]
Affected Services: [list]
Current Impact: [description]
Next Update: [estimated time]
```

**Update Notification:**
```
INCIDENT UPDATE
Incident ID: [ID]
Time: [timestamp]
Status: [INVESTIGATING|IDENTIFIED|MONITORING|RESOLVED]
Update: [what's happening]
ETA: [if available]
```

**Resolution Notification:**
```
INCIDENT RESOLVED
Incident ID: [ID]
Time: [timestamp]
Resolution: [what was fixed]
Downtime: [duration]
Post-Mortem: [scheduled time]
```

---

## Testing and Validation

### Testing Schedule

| Test Type | Frequency | Owner |
|-----------|-----------|--------|
| Backup Restoration | Weekly | DevOps |
| Failover Drill | Monthly | DevOps |
| Full DRP Test | Quarterly | DevOps + Management |
| Security Audit | Quarterly | Security Team |

### Validation Checklist

After recovery:
- [ ] Database integrity verified
- [ ] All services running
- [ ] Trading agents operational
- [ ] API endpoints responding
- [ ] Dashboard accessible
- [ ] No data loss detected
- [ ] No unauthorized activity
- [ ] Performance baseline met

---

## Contacts

### Emergency Contacts

| Role | Name | Phone | Email |
|-------|------|-------|-------|
| DevOps Lead | [Name] | +1-XXX-XXX-XXXX | devops@trading-ai.com |
| Security Lead | [Name] | +1-XXX-XXX-XXXX | security@trading-ai.com |
| Engineering Lead | [Name] | +1-XXX-XXX-XXXX | eng@trading-ai.com |
| Management | [Name] | +1-XXX-XXX-XXXX | mgmt@trading-ai.com |

### Exchange Support Contacts

| Exchange | Support Email | Phone | Status Page |
|----------|---------------|-------|-------------|
| Binance | support@binance.com | N/A | https://status.binance.com/ |
| Coinbase | support@coinbase.com | N/A | https://status.coinbase.com/ |
| Alpaca | support@alpaca.markets | N/A | https://status.alpaca.markets/ |
| Uniswap | N/A | N/A | N/A (DAO) |

### Service Providers

| Service | Contact | Priority |
|----------|----------|-----------|
| AWS Support | N/A | 1 |
| DNS Provider | N/A | 2 |
| ISP | N/A | 1 |

---

## Appendix A: Command Reference

### Database Commands

```bash
# Backup database
python -m src.infrastructure.backup_manager daily

# Manual backup
python -m src.infrastructure.backup_manager manual

# Restore backup
python -m src.infrastructure.backup_manager restore <backup_id>

# List backups
python -m src.infrastructure.backup_manager list

# Verify backup
python -m src.infrastructure.backup_manager verify <backup_id>

# Cleanup old backups
python -m src.infrastructure.backup_manager cleanup
```

### Trading Agent Commands

```bash
# Start live trading
python start.py --mode live

# Emergency stop
python -m src.autonomous_agent.trading_agent --emergency-stop

# Check status
python start.py --status

# List agents
python start.py --agents

# Pause specific agent
curl -X POST http://localhost:8000/api/agents/<agent_id>/pause
```

### System Health Commands

```bash
# Run pre-launch validator
python -m src.deployment.prelaunch_validator

# Run security audit
python -m src.security.security_audit

# Check docker services
docker-compose ps

# View logs
docker-compose logs -f <service>
tail -f logs/trading.log
```

---

## Appendix B: Change Log

| Date | Version | Changes |
|-------|---------|----------|
| 2026-03-08 | 1.0 | Initial DRP document |

---

**This document should be reviewed and updated at least quarterly or after any major incident.**
