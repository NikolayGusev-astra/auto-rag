#!/usr/bin/env python3
"""Golden set eval using UnifiedSearcher (auto-detect ZVec/Chroma)."""

import sys
import json
import time
sys.path.insert(0, "rag_core")
from unified_searcher import UnifiedSearcher
from rag_core.dcd_router import classify

SEARCHER = UnifiedSearcher()

GOLDEN_SET = [
    ("как настроить postgresql streaming replication", "database"),
    ("rust ownership borrow checker", "software-dev"),
    ("шрус ford explorer 2016 партномер", "ford-club"),
    ("nginx reverse proxy ssl letsencrypt", "linux-admin"),
    ("wireguard vpn config", "networking"),
    ("systemd service unit timer", "linux-admin"),
    ("docker compose up production", "devops"),
    ("kubernetes deployment helm chart", "devops"),
    ("terraform plan apply", "devops"),
    ("prometheus alert rule", "monitoring"),
    ("zfs pool create raidz2", "storage"),
    ("stalwart smtp dkim spf", "email"),
    ("linux kernel sysctl tuning", "kernel"),
    ("bash script find grep awk", "scripting"),
    ("adr architecture decision record", "docs"),
    ("postgresql vacuum analyze autovacuum", "database"),
    ("xray reality vless config", "networking"),
    ("proxmox lxc container backup restore", "virtualization"),
    ("kvm gpu passthrough vfio iommu", "virtualization"),
    ("zfs snapshot send receive incremental", "storage"),
    ("postfix dovecot sieve filter", "email"),
    ("grafana dashboard panel variable templating", "monitoring"),
    ("linux bridge vlan trunk access", "networking"),
    ("systemd override drop-in service", "linux-admin"),
]

KEY_TERMS = {
    "как настроить postgresql streaming replication": ["postgresql", "streaming", "replication", "wal", "walreceiver", "primary", "standby"],
    "rust ownership borrow checker": ["rust", "ownership", "borrow", "borrow checker", "lifetime", "move", "reference"],
    "шрус ford explorer 2016 партномер": ["шрус", "ford", "explorer", "партномер", "catalog", "oem"],
    "nginx reverse proxy ssl letsencrypt": ["nginx", "reverse proxy", "ssl", "letsencrypt", "certbot", "proxy_pass"],
    "wireguard vpn config": ["wireguard", "vpn", "config", "peer", "endpoint", "allowedips"],
    "systemd service unit timer": ["systemd", "service", "unit", "timer", "oncalendar", "systemctl"],
    "docker compose up production": ["docker", "compose", "production", "deploy", "stack", "yml"],
    "kubernetes deployment helm chart": ["kubernetes", "deployment", "helm", "chart", "pod", "replicaset"],
    "terraform plan apply": ["terraform", "plan", "apply", "state", "provider", "resource"],
    "prometheus alert rule": ["prometheus", "alert", "rule", "alertmanager", "expr", "for"],
    "zfs pool create raidz2": ["zfs", "pool", "create", "raidz2", "vdev", "zpool"],
    "stalwart smtp dkim spf": ["stalwart", "smtp", "dkim", "spf", "dmarc", "mx"],
    "linux kernel sysctl tuning": ["linux", "kernel", "sysctl", "tuning", "proc", "sys"],
    "bash script find grep awk": ["bash", "script", "find", "grep", "awk", "sed"],
    "adr architecture decision record": ["adr", "architecture decision record", "decision", "context", "consequence"],
    "postgresql vacuum analyze autovacuum": ["postgresql", "vacuum", "analyze", "autovacuum", "freeze", "bloat"],
    "xray reality vless config": ["xray", "reality", "vless", "config", "uuid", "flow"],
    "proxmox lxc container backup restore": ["proxmox", "lxc", "container", "backup", "restore", "vzdump"],
    "kvm gpu passthrough vfio iommu": ["kvm", "gpu", "passthrough", "vfio", "iommu", "vfio-pci"],
    "zfs snapshot send receive incremental": ["zfs", "snapshot", "send", "receive", "incremental", "zfs send"],
    "postfix dovecot sieve filter": ["postfix", "dovecot", "sieve", "filter", "spam", "mail"],
    "grafana dashboard panel variable templating": ["grafana", "dashboard", "panel", "variable", "templating", "query"],
    "linux bridge vlan trunk access": ["linux", "bridge", "vlan", "trunk", "access", "802.1q"],
    "systemd override drop-in service": ["systemd", "override", "drop-in", "service", "systemctl edit"],
}

def main():
    print("=" * 80)
    print("CUSTOM GOLDEN SET EVALUATION — UnifiedSearcher (auto ZVec/Chroma)")
    print("=" * 80)
    
    domain_correct = 0
    fact_correct = 0
    total = len(GOLDEN_SET)
    results = []
    
    for i, (query, expected_domain) in enumerate(GOLDEN_SET, 1):
        t0 = time.time()
        
        # DCD classification
        dcd_result = classify(query)
        domain = dcd_result.get("domain", "")
        domain_ok = domain == expected_domain
        
        # Search
        hits = SEARCHER.search(query, topk=5)
        t1 = time.time()
        
        # Fact recall
        expected_terms = KEY_TERMS.get(query, [])
        hit_texts = " ".join(h.get("content", "") + " " + h.get("heading", "") + " " + h.get("title", "") for h in hits).lower()
        found_terms = [t for t in expected_terms if t.lower() in hit_texts]
        fact_ok = len(found_terms) >= max(1, len(expected_terms) // 2)
        
        if domain_ok:
            domain_correct += 1
        if fact_ok:
            fact_correct += 1
        
        status = "✅" if (domain_ok and fact_ok) else "❌"
        print(f"{status} [{i:2d}/{total}] {query[:55]:55s} dom={'✓' if domain_ok else '✗'}({domain}) fact={'✓' if fact_ok else '✗'}({len(found_terms)}/{len(expected_terms)}) [{t1-t0:.2f}s]")
        
        results.append({
            "query": query,
            "expected_domain": expected_domain,
            "predicted_domain": domain,
            "domain_ok": domain_ok,
            "fact_ok": fact_ok,
            "found_terms": found_terms,
            "latency_ms": round((t1 - t0) * 1000, 1),
            "top_score": hits[0]["score"] if hits else 0,
        })
    
    print("=" * 80)
    print(f"Total:         {total}")
    print(f"Domain acc:    {domain_correct}/{total} = {domain_correct/total*100:.1f}%")
    print(f"Fact recall:   {fact_correct}/{total} = {fact_correct/total*100:.1f}%")
    print(f"Backend:       {SEARCHER.backend}")
    print(f"Stats:         {SEARCHER.get_stats()}")
    print("=" * 80)
    
    report = {
        "backend": SEARCHER.backend,
        "stats": SEARCHER.get_stats(),
        "total": total,
        "domain_accuracy": domain_correct / total,
        "fact_recall": fact_correct / total,
        "results": results,
    }
    with open("golden_report_unified.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to golden_report_unified.json")

if __name__ == "__main__":
    main()