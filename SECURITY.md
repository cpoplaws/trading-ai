# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it to us privately:

1. **Do NOT** open a public issue
2. Email: [your-security-email@example.com]
3. Or use GitHub's private vulnerability reporting: [Security Advisories](https://github.com/cpoplaws/trading-ai/security/advisories/new)

We will respond within 48 hours and work with you to address the issue.

---

## Security Updates

### Current Status

✅ **Last Security Audit**: 2026-02-16
✅ **Known Vulnerabilities**: 0 high-severity (after applying latest updates)
⚠️ **Action Required**: Apply security updates (see below)

### Quick Fix

Apply all security updates immediately:

```bash
# Automated update (recommended)
./scripts/apply_security_updates.sh

# Or manual update
pip install --upgrade -r requirements-security-update.txt
pytest tests/  # Verify nothing broke
```

### Recent Security Updates (2026-02-16)

Fixed **12 vulnerabilities** (7 high, 5 moderate):

#### High Severity (Fixed)
- ✅ urllib3: Request smuggling, header injection, TLS bypass
- ✅ cryptography: Weak cipher modes
- ✅ pillow: Buffer overflow in image processing
- ✅ aiohttp: HTTP request smuggling
- ✅ jinja2: Server-side template injection
- ✅ werkzeug: Debug mode code execution

#### Moderate Severity (Fixed)
- ✅ certifi: Outdated SSL certificates
- ✅ pyyaml: Unsafe deserialization
- ✅ sqlalchemy: SQL injection vectors
- ✅ fastapi: Validation bypass
- ✅ starlette: Path traversal

**Details**: See [Security Update Documentation](./docs/SECURITY_UPDATE_2026-02-16.md)

---

## Automated Security Scanning

### GitHub Actions

We run automated security scans:
- **On every push** to main/develop
- **On all pull requests**
- **Weekly** (Mondays at 9 AM UTC)

Scans include:
1. **Safety** - PyPA Advisory Database
2. **pip-audit** - OSV Database
3. **Bandit** - Static security analysis
4. **Dependabot** - GitHub native scanning

### Local Security Checks

Run security checks locally before committing:

```bash
# Install security tools
pip install safety pip-audit bandit

# Check for known vulnerabilities
safety check

# Audit dependencies (alternative scanner)
pip-audit

# Check code for security issues
bandit -r src/

# All-in-one check
./scripts/run_security_checks.sh
```

---

## Security Best Practices

### 1. Dependency Management

✅ **DO**:
- Use `requirements-secure.txt` for production
- Update dependencies monthly minimum
- Apply security patches within 48 hours
- Pin major versions (e.g., `package>=1.2.0,<2.0`)

❌ **DON'T**:
- Use unpinned dependencies (`package>=1.0.0`)
- Ignore Dependabot alerts
- Install packages without verifying source
- Use `pip install --trusted-host` in production

### 2. Secrets Management

✅ **DO**:
- Use environment variables for secrets
- Use `.env` files (never commit them!)
- Use secret management services (AWS Secrets Manager, etc.)
- Rotate API keys regularly

❌ **DON'T**:
- Hardcode API keys in source code
- Commit `.env` files to git
- Share secrets in plain text (Slack, email)
- Use default passwords

### 3. API Security

✅ **DO**:
- Use HTTPS for all API calls
- Validate all user inputs
- Implement rate limiting
- Use API keys with minimal permissions
- Enable 2FA on all accounts

❌ **DON'T**:
- Make API calls over HTTP
- Trust user input without validation
- Use admin API keys for regular operations
- Disable SSL verification (`verify=False`)

### 4. Trading Security

⚠️ **CRITICAL**:
- **Paper trade first** - Never test strategies with real money
- **Start small** - Use minimal capital for initial live trading
- **Set limits** - Configure max position size, daily loss limits
- **Monitor actively** - Set up alerts for unusual activity
- **Backup everything** - Database, configurations, logs

### 5. Infrastructure Security

✅ **DO**:
- Keep Docker images updated
- Use non-root users in containers
- Enable firewall on servers
- Regular security audits
- Monitor logs for suspicious activity

❌ **DON'T**:
- Expose databases to public internet
- Run services as root
- Use default credentials
- Disable security features for "convenience"

---

## Vulnerability Disclosure Timeline

When we discover or are notified of a vulnerability:

1. **Day 0**: Vulnerability reported/discovered
2. **Day 1**: Confirm vulnerability and assess severity
3. **Day 1-2**: Develop and test fix
4. **Day 2**: Deploy fix to production (for high/critical)
5. **Day 7**: Public disclosure (after users have time to update)

For critical vulnerabilities (RCE, data exposure), we deploy fixes immediately.

---

## Security Contacts

- **Security Team**: [security-team-email]
- **On-Call (Critical)**: [on-call-phone]
- **GitHub**: https://github.com/cpoplaws/trading-ai/security

---

## Compliance

This project follows:
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## Security Checklist for Contributors

Before submitting a pull request:

- [ ] No secrets in code or config files
- [ ] All inputs validated
- [ ] Dependencies up to date (`pip list --outdated`)
- [ ] Security checks pass (`safety check`, `bandit -r src/`)
- [ ] No SQL queries without parameterization
- [ ] No eval(), exec(), or pickle.load() on untrusted data
- [ ] HTTPS used for all external API calls
- [ ] Tests include security edge cases

---

## Additional Resources

- [Security Update Guide](./docs/SECURITY_UPDATE_2026-02-16.md)
- [Dependency Update Plan](./docs/DEPENDENCY_UPDATE_PLAN.md)
- [Error Recovery Playbook](./docs/ERROR_RECOVERY_PLAYBOOK.md)
- [Troubleshooting Guide](./docs/TROUBLESHOOTING_GUIDE.md)

---

**Last Updated**: 2026-02-16
**Next Review**: 2026-03-16
