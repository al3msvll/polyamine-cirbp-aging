"""
Polyamine-CIRBP gene panel used by the age clocks (103 unique genes, 6 functional groups;
hypusination group includes eIF5A polyproline targets). Sub-score, machinery and
inflammation gene sets are defined below.
"""

PANEL = {
    'Polyamine': ['ODC1','AMD1','SRM','SMS','SAT1','PAOX','SMOX'],
    'Hypusination': ['DHPS','DOHH','EIF5A','EIF5A2','PTPN1','ATF4','ATF5','NFKB2','RELB',
                     'SMAD2','TGFBR2','CHGA','COL1A1','COL3A1','FN1','KRAS','MAPK1',
                     'MAP2K1','RHOA','RAC1','CDC42','RAB7A'],
    'CIRBP_targets': ['CIRBP','HMGB1','VEGFA','TNF','IL6','CXCL8','TP53','CCNB1','CCND1',
                      'MYC','BCL2','BAX','GADD45B','ATF4','HIF1A','CST3','MMP9','NFE2L2',
                      'XIAP','BIRC5','E2F1','RPS3','HSPA1A','HSPB1','ATF3','DUSP1',
                      'FOS','JUN','EGR1','NR4A1','CDKN1A','MDM2','RAB7A'],
    'SASP': ['IL6','CXCL8','IL1A','IL1B','TNF','CCL2','CCL20','CXCL1','CXCL2','CXCL10',
             'MMP1','MMP3','MMP10','MMP12','TIMP1','IGFBP3','IGFBP7','VEGFA','HGF',
             'PLAUR','SERPINE1','SERPINB2','ICAM1','FAS','FASLG','ANGPTL4','PTGS2',
             'NPPB','NRG1','LTA'],
    'NFkB': ['NFKB1','NFKB2','RELA','RELB','REL','NFKBIA','NFKBIB','TLR2','TLR4',
             'MYD88','IRAK1','IRAK4','TRAF6','TNFAIP3','BIRC3'],
    'CellCycle_Senescence': ['CDKN1A','CDKN2A','TP53','MKI67','LMNB1','RB1']
}

# Build unique gene list and group mapping
_all = [(g, grp) for grp, genes in PANEL.items() for g in genes]
UNIQUE_GENES = []
_seen = set()
GENE_GROUPS = {}
for g, grp in _all:
    if g not in _seen:
        _seen.add(g)
        UNIQUE_GENES.append(g)
        GENE_GROUPS[g] = grp

# Count duplicates
_gene_count = {}
for g, _ in _all:
    _gene_count[g] = _gene_count.get(g, 0) + 1
DUPLICATE_GENES = {g for g, c in _gene_count.items() if c > 1}

# Sub-score gene groups (non-overlapping for scoring)
POLYAMINE_GENES = ['ODC1','AMD1','SRM','SMS','SAT1','PAOX','SMOX']
HYPUSINATION_GENES = ['DHPS','DOHH','EIF5A','EIF5A2']
CIRBP_MODULE_GENES = ['CIRBP','HMGB1','TNF','IL6','CXCL8','TP53','CDKN1A','FOS','JUN','EGR1','NR4A1','DUSP1','ATF3','HIF1A','NFE2L2']
SASP_GENES = ['IL6','CXCL8','IL1A','IL1B','TNF','CCL2','CCL20','CXCL1','CXCL2','CXCL10',
              'MMP1','MMP3','MMP10','MMP12','TIMP1','IGFBP3','IGFBP7','VEGFA','PTGS2','ICAM1','FAS','SERPINE1']
MACHINERY_GENES = POLYAMINE_GENES + HYPUSINATION_GENES
INFLAMMATION_GENES = SASP_GENES + ['NFKB1','NFKB2','RELA','RELB','REL','NFKBIA','TLR4','MYD88','IRAK4','TRAF6','TNFAIP3']

if __name__ == '__main__':
    print(f"Panel: {len(UNIQUE_GENES)} unique genes, {len(DUPLICATE_GENES)} appear in multiple groups: {sorted(DUPLICATE_GENES)}")
    print(f"Groups: {', '.join(f'{k}({len(v)})' for k,v in PANEL.items())}")
    print(f"Sub-scores: PPAS={len(POLYAMINE_GENES)}, Hypusination={len(HYPUSINATION_GENES)}, CIRBP-module={len(CIRBP_MODULE_GENES)}, SASP={len(SASP_GENES)}")
    print(f"Machinery={len(MACHINERY_GENES)}, Inflammation={len(INFLAMMATION_GENES)}")
