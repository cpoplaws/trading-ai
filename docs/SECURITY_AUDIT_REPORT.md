# Security Audit Report

**Date**: February 16, 2026
**Audit Type**: Dependency Vulnerability Scan
**Tool**: pip-audit 2.9.0
**Scope**: Python dependencies in requirements.txt and requirements-crypto.txt

## Executive Summary

Security audit identified **12 known vulnerabilities** across **3 packages**:
- **7 High Severity** vulnerabilities in Keras
- **4 High Severity** vulnerabilities in urllib3
- **1 Moderate Severity** vulnerability in Pillow

**Risk Level**: HIGH
**Recommended Action**: IMMEDIATE UPDATE REQUIRED

## Vulnerability Details

### 1. urllib3 (Current: 1.26.20)

**Package**: urllib3
**Current Version**: 1.26.20
**Latest Safe Version**: 2.6.3+
**Total Vulnerabilities**: 4

#### CVE-2025-50181 (GHSA-pq67-6m6q-mj2v)
- **Severity**: HIGH
- **Fix Version**: 2.5.0+
- **Description**: Circuit breaker OPEN vulnerability - redirect handling bypass
- **Impact**: SSRF and open redirect vulnerabilities not properly mitigated
- **Exploitation**: Applications attempting to disable redirects at PoolManager level remain vulnerable

#### CVE-2025-66418 (GHSA-gm62-xv2j-4w53)
- **Severity**: HIGH
- **Fix Version**: 2.6.0+
- **Description**: Unbounded HTTP encoding chain decompression
- **Impact**: High CPU usage and massive memory allocation
- **Exploitation**: Malicious server can insert unlimited compression steps leading to DoS

#### CVE-2025-66471 (GHSA-2xpw-w6gg-jr37)
- **Severity**: HIGH
- **Fix Version**: 2.6.0+
- **Description**: Decompression bomb in streaming API
- **Impact**: Excessive resource consumption (CPU and memory)
- **Exploitation**: Small highly compressed data causes full decompression in single operation

#### CVE-2026-21441 (GHSA-38jv-5279-wg99)
- **Severity**: HIGH
- **Fix Version**: 2.6.3+
- **Description**: Decompression of redirect responses when preload_content=False
- **Impact**: No safeguard against decompression bombs in redirects
- **Exploitation**: Malicious server can trigger excessive resource consumption

**Remediation**: Upgrade to urllib3>=2.6.3

---

### 2. Pillow (Current: 11.3.0)

**Package**: Pillow
**Current Version**: 11.3.0
**Latest Safe Version**: 12.1.1+
**Total Vulnerabilities**: 1

#### CVE-2026-25990 (GHSA-cfh3-3jmp-rvhc)
- **Severity**: MODERATE
- **Fix Version**: 12.1.1+
- **Description**: Out-of-bounds write in PSD image loading
- **Impact**: Memory corruption, potential arbitrary code execution
- **Exploitation**: Specially crafted PSD image triggers buffer overflow
- **Workaround**: Use `formats` parameter in `Image.open()` to block PSD files

**Remediation**: Upgrade to Pillow>=12.1.1

---

### 3. Keras (Current: 3.10.0)

**Package**: Keras (TensorFlow)
**Current Version**: 3.10.0
**Latest Safe Version**: 3.13.1+ (3.13.2 recommended, one vuln unpatched)
**Total Vulnerabilities**: 7 (1 unpatched)

#### CVE-2025-8747 (GHSA-c9rc-mg46-23w3)
- **Severity**: HIGH
- **Fix Version**: 3.11.0+
- **Description**: Bypass of CVE-2025-1550 mitigation in model loading
- **Impact**: Arbitrary file overwrite, potential RCE
- **Exploitation**: Crafted .keras model bypasses safe_mode protection
- **Attack Vector**: Reusing internal Keras functions like `keras.utils.get_file()`

#### CVE-2025-9906 (GHSA-36fq-jgmw-4r9c)
- **Severity**: HIGH
- **Fix Version**: 3.11.0+
- **Description**: Arbitrary code execution when loading .keras archives
- **Impact**: Full system compromise
- **Exploitation**: Archive invokes `enable_unsafe_deserialization()` before layer loading
- **Attack Vector**: Pickle deserialization of Lambda layer functions

#### CVE-2025-9905 (GHSA-36rr-ww3j-vrjv)
- **Severity**: HIGH
- **Fix Version**: 3.11.3+
- **Description**: safe_mode silently ignored for .h5/.hdf5 files
- **Impact**: Arbitrary code execution
- **Exploitation**: Lambda layers in .h5 files execute during model loading
- **Attack Vector**: Exec() in Lambda function, triggers on load

#### CVE-2025-12058 (GHSA-mq84-hjqx-cwf2)
- **Severity**: HIGH
- **Fix Version**: 3.12.0+
- **Description**: Arbitrary local file read and SSRF via StringLookup layer
- **Impact**: Information disclosure, SSRF attacks
- **Exploitation**: Malicious .keras file with StringLookup vocabulary path
- **Attack Vector**: Reads local files or performs SSRF via GCS/HDFS handlers

#### CVE-2025-12060 (GHSA-hjqc-jx6g-rwp9)
- **Severity**: HIGH
- **Fix Version**: 3.12.0+
- **Description**: Directory traversal in `keras.utils.get_file()`
- **Impact**: Arbitrary file write outside intended directory
- **Exploitation**: PATH_MAX symlink resolution bug bypasses path filtering
- **Attack Vector**: Malicious tar archive with deep symlink chains

#### CVE-2026-0897 (GHSA-xfhx-r7ww-5995)
- **Severity**: MODERATE
- **Fix Version**: 3.13.1+
- **Description**: DoS via memory exhaustion in HDF5 weight loading
- **Impact**: Denial of Service, Python interpreter crash
- **Exploitation**: Crafted .keras archive with extremely large dataset shapes
- **Attack Vector**: model.weights.h5 with malicious shape declarations

#### CVE-2026-1669 (GHSA-gfmx-qqqh-f38q) ⚠️ UNPATCHED
- **Severity**: HIGH
- **Fix Version**: None available
- **Description**: Arbitrary file read via HDF5 external dataset references
- **Impact**: Local file disclosure
- **Exploitation**: Crafted .keras model with HDF5 external references
- **Workaround**: Validate models from trusted sources only

**Remediation**: Upgrade to Keras>=3.13.1 (awaiting patch for CVE-2026-1669)

---

## Impact Assessment

### Critical Attack Vectors

1. **Model Poisoning**
   - Malicious ML models can execute arbitrary code on load
   - Affects: Research environments, automated training systems
   - Risk: CRITICAL

2. **Data Exfiltration**
   - Arbitrary file read vulnerabilities expose sensitive data
   - Affects: Systems loading untrusted models
   - Risk: HIGH

3. **Denial of Service**
   - Decompression bombs and memory exhaustion
   - Affects: Production ML services
   - Risk: HIGH

4. **SSRF Attacks**
   - Internal network scanning via model loading
   - Affects: Cloud-deployed ML systems
   - Risk: HIGH

### Affected Components

- ✅ Trading Agent (uses Keras for ML models)
- ✅ REST API (uses urllib3 via requests)
- ✅ Data Processing (uses Pillow for chart generation)
- ✅ Web Dashboard (uses Streamlit which depends on Pillow)

### Business Impact

- **Confidentiality**: HIGH - Arbitrary file read, SSRF
- **Integrity**: HIGH - Arbitrary code execution
- **Availability**: MODERATE - DoS attacks possible

## Remediation Plan

### Phase 1: Immediate Actions (Within 24 hours)

1. **Update Critical Dependencies**
   ```bash
   pip install --upgrade urllib3>=2.6.3
   pip install --upgrade pillow>=12.1.1
   pip install --upgrade keras>=3.13.1
   ```

2. **Verify Compatibility**
   ```bash
   pytest tests/
   python -m src.utils.structured_logging  # Test logging
   python -m src.api.main  # Test API
   ```

3. **Deploy to Test Environment**
   - Run full integration tests
   - Verify all services start correctly
   - Test ML model loading (safe models only)

### Phase 2: Validation (Within 48 hours)

1. **Re-run Security Audit**
   ```bash
   pip-audit -r requirements.txt
   pip-audit -r requirements-crypto.txt
   ```

2. **Test Critical Paths**
   - Model loading and inference
   - API request handling
   - Image processing in dashboard

3. **Monitor for Issues**
   - Check error logs
   - Monitor performance metrics
   - Verify no breaking changes

### Phase 3: Hardening (Within 1 week)

1. **Implement Additional Controls**
   - Model validation before loading
   - Request size limits for API
   - Resource limits for ML operations

2. **Update Documentation**
   - Security best practices
   - Model loading guidelines
   - Dependency update procedures

3. **Set Up Automated Scanning**
   - GitHub Dependabot alerts
   - Weekly security audits
   - Automated dependency updates

## Updated Requirements

See `requirements-updated.txt` and `requirements-crypto-updated.txt`

### Key Changes

- urllib3: 1.26.20 → 2.6.3
- pillow: 11.3.0 → 12.1.1
- keras: 3.10.0 → 3.13.1 (via tensorflow 2.20.0 → latest)

### Breaking Changes

**urllib3 2.x Changes:**
- Requires OpenSSL 1.1.1+ (LibreSSL not supported)
- Some API changes from 1.x to 2.x
- May affect code using advanced urllib3 features

**Mitigation**: Most code uses urllib3 via requests library, minimal impact expected

## Monitoring and Prevention

### Continuous Security

1. **Enable Dependabot**
   ```yaml
   # .github/dependabot.yml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       open-pull-requests-limit: 10
   ```

2. **Pre-commit Security Checks**
   ```yaml
   # .pre-commit-config.yaml
   - repo: https://github.com/pyupio/safety
     rev: 2.3.5
     hooks:
       - id: safety
   ```

3. **CI/CD Integration**
   ```yaml
   # .github/workflows/security.yml
   - name: Security audit
     run: |
       pip install pip-audit
       pip-audit -r requirements.txt --format json
   ```

### Security Best Practices

1. **Model Loading**
   - ✅ Always use `safe_mode=True` for Keras models
   - ✅ Validate model sources (checksums, signatures)
   - ✅ Load models from trusted repositories only
   - ✅ Scan models before deployment

2. **API Security**
   - ✅ Implement rate limiting (already done)
   - ✅ Validate all inputs
   - ✅ Use circuit breakers (already done)
   - ✅ Monitor for anomalies

3. **Dependency Management**
   - ✅ Pin exact versions in production
   - ✅ Test updates in staging first
   - ✅ Review security advisories weekly
   - ✅ Automate dependency updates

## Verification Checklist

After applying updates:

- [ ] All dependencies updated to safe versions
- [ ] Security audit shows 0 vulnerabilities
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Application starts without errors
- [ ] ML models load correctly
- [ ] API responds normally
- [ ] Dashboard renders correctly
- [ ] No performance degradation
- [ ] Documentation updated
- [ ] Team notified of changes
- [ ] Deployment schedule confirmed

## References

- [pip-audit documentation](https://pypi.org/project/pip-audit/)
- [urllib3 security advisories](https://github.com/urllib3/urllib3/security/advisories)
- [Pillow security advisories](https://github.com/python-pillow/Pillow/security/advisories)
- [Keras security advisories](https://github.com/keras-team/keras/security/advisories)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)

---

**Report Generated**: 2026-02-16 00:30:00 UTC
**Next Audit**: 2026-02-23 (weekly schedule)
**Contact**: security@trading-ai.local
