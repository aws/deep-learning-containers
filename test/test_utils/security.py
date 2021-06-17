from enum import IntEnum


class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ECRScanFalseAlarm:
    def __init__(self, package, status):
        self.package = package
        self.severity = status["Severity"]
        self.cve = status["CVE"]
        self.reason = status["Reason"]
