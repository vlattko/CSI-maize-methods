"""Nested registered-index comparisons with an explicit biological-feature whitelist."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import prediction as p

CHANNELS = ('rgb_R','rgb_G','rgb_B','noir_R','noir_G','noir_B')
ND_IDS = tuple('ND_'+a+'__'+b for i,a in enumerate(CHANNELS) for b in CHANNELS[i+1:])
SIX_IDS = ('CAMSUM6','GREENSHARE6','CHROMDIST6','GMEAN6','PRODUCT4','GLICONTRAST6')
INDEX_IDS = ND_IDS+('ExG','ExGR','GLI','VARI','MGRVI','RGBVI')+SIX_IDS
CROSS_IDS = tuple(i for i in ND_IDS if 'rgb_' in i and 'noir_' in i)+SIX_IDS
INDEX_COLUMNS = tuple(f'paired_index__{i}__{s}' for i in INDEX_IDS for s in ('median','iqr'))
CROSS_COLUMNS = tuple(f'paired_index__{i}__{s}' for i in CROSS_IDS for s in ('median','iqr'))
DN_COLUMNS = tuple(f'paired_dn__{c}__{s}' for c in CHANNELS for s in ('median','iqr'))


def designs(rows,mode):
    base = p.feature_sets(rows,True,mode,'paired_primary')
    meta,meta_names = base['metadata_ridge']
    traits,trait_names = base['combined']

    def matrix(names):
        return np.array([[p.number(r[n]) for n in names] for r in rows])

    indices,cross,dns = (matrix(n) for n in (INDEX_COLUMNS,CROSS_COLUMNS,DN_COLUMNS))
    return dict(metadata_ridge=(meta,meta_names),
        sam_traits=(traits,trait_names),
        registered_indices=(indices,list(INDEX_COLUMNS)),
        cross_camera_indices=(cross,list(CROSS_COLUMNS)),
        sam_traits_plus_indices=(np.hstack([traits,indices]),trait_names+list(INDEX_COLUMNS)),
        metadata_plus_indices=(np.hstack([meta,indices]),meta_names+list(INDEX_COLUMNS)),
        registered_camera_dns=(dns,list(DN_COLUMNS)),
        camera_dns_plus_cross_indices=(np.hstack([dns,cross]),list(DN_COLUMNS+CROSS_COLUMNS)),
        metadata_plus_dns=(np.hstack([meta,dns]),meta_names+list(DN_COLUMNS)),
        metadata_plus_dns_plus_cross_indices=(np.hstack([meta,dns,cross]),
                                             meta_names+list(DN_COLUMNS+CROSS_COLUMNS)))


def assert_availability(rows,prefix,definitions):
    for row in rows:
        for definition in definitions:
            stem = f'{prefix}__{definition}__'
            count = int(row[stem+'n_views'])
            eligible = p.truth(row[stem+'eligible'])
            finite = [np.isfinite(p.number(row[stem+s])) for s in ('median','iqr')]
            if count not in range(5) or eligible!=(count>=2) or finite!=[eligible,eligible]:
                raise ValueError('Index/DN aggregation availability contract mismatch')


def ready(args):
    for path in (args.run/'extraction'/'summary.json',args.index_manifest,args.dn_manifest):
        if not path.exists() or json.loads(path.read_text(encoding='utf-8'))['status']!='completed':
            return False
    completed_images=args.run/'prediction'/'summary_images.json'
    if completed_images.exists() and json.loads(completed_images.read_text(encoding='utf-8'))['status']=='completed':
        return True
    phase = args.run/'prediction'/'finalizer_status.json'
    if not phase.exists():
        return False
    return json.loads(phase.read_text(encoding='utf-8'))['status'] in ('completed','failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('intake','run','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--index-manifest',type=Path)
    parser.add_argument('--dn-manifest',type=Path)
    args = parser.parse_args()
    pair = args.run/'pairing'/'full'
    args.index_manifest = args.index_manifest or pair/'session_descriptors_manifest.json'
    args.dn_manifest = args.dn_manifest or pair/'session_camera_dn_manifest.json'
    destination = args.run/'prediction'
    state = destination/'indices_status.json'
    while not ready(args):
        if not args.watch:
            raise ValueError('Completed extraction, pairing, DN controls and image phase required')
        p.save_json(state,dict(status='waiting_for_pairing_and_image_phase'))
        time.sleep(20)
    p.save_json(state,dict(status='running'))
    try:
        sessions_path = args.run/'extraction'/'session_traits.csv'
        p.validate_extraction(sessions_path)
        reference_path = args.intake/'research_audit'/'reference_master.csv'
        references = p.read_keyed(reference_path)
        sessions = p.read_keyed(sessions_path)
        index_path = pair/'session_descriptors.csv'
        dn_path = pair/'session_camera_dn.csv'
        index_manifest = json.loads(args.index_manifest.read_text(encoding='utf-8'))
        dn_manifest = json.loads(args.dn_manifest.read_text(encoding='utf-8'))
        if index_manifest['session_descriptors_sha256']!=p.digest(index_path):
            raise ValueError('Index table hash mismatch')
        if set(index_manifest['feature_columns'])!=set(INDEX_COLUMNS):
            raise ValueError('Unexpected index feature semantics')
        if dn_manifest['session_camera_dn_sha256']!=p.digest(dn_path):
            raise ValueError('DN table hash mismatch')
        if set(dn_manifest['feature_columns'])!=set(DN_COLUMNS):
            raise ValueError('Unexpected DN feature semantics')
        indices,dns = p.read_keyed(index_path),p.read_keyed(dn_path)
        assert_availability(indices,'paired_index',INDEX_IDS)
        assert_availability(dns,'paired_dn',CHANNELS)
        expected = {p.key(r) for r in references}
        if len(references)!=360 or any({p.key(r) for r in rows}!=expected
                                      for rows in (sessions,indices,dns)):
            raise ValueError('All tables must retain exactly360 assay keys')
        maps = [{p.key(r):r for r in rows} for rows in (sessions,indices,dns)]
        joined = [{**r,**maps[0][p.key(r)],**maps[1][p.key(r)],**maps[2][p.key(r)]}
                  for r in references]
        chosen = p.cohort_indices(joined,'paired_primary')
        rows = [joined[i] for i in chosen]
        p.save_csv(destination/'selection_indices.csv',[dict(**{k:r[k] for k in p.KEYS},
            primary_paired_eligible=bool(i in chosen),
            **{name:r[name] for name in r if name.startswith(('paired_index__','paired_dn__'))
               and name.endswith(('__n_views','__eligible','__missing_reason'))})
            for i,r in enumerate(joined)])
        hashes = dict(source=p.digest(__file__),prediction_operator=p.digest(p.__file__),
            references=p.digest(reference_path),sessions=p.digest(sessions_path),
            indices=p.digest(index_path),dns=p.digest(dn_path),
            index_manifest=p.digest(args.index_manifest),dn_manifest=p.digest(args.dn_manifest))
        config = dict(hashes=hashes,reference_rows=360,primary_paired_rows=len(rows),
            index_columns=INDEX_COLUMNS,cross_columns=CROSS_COLUMNS,dn_columns=DN_COLUMNS,
            alphas=p.ALPHAS,seeds=p.SEEDS,
            limitations=['Indices are deterministic functions of six recorded channels, not new bands.',
            'The matched DN control uses common overlap; same-camera index families use their own interiors.',
            'Per-channel DN validity guards and pairwise formula guards can retain different pixel subsets.',
            'Incremental held-out performance is a representation comparison, not calibrated physiology.',
            'Availability/QC/registration fit/control fields are not predictors.'])
        (destination/'execution_source_indices.py').write_bytes(Path(__file__).read_bytes())
        domains = [('assay_'+label,(target,),np.array([[p.number(r[target])] for r in rows]),{})
                   for target,label in zip(p.TARGETS,('FvFm','PiAbs','NIR_NDVI'))]
        for band in ('UVIS','NIR'):
            waves,values,flags = p.read_spectra(args.intake,band)
            config['hashes'][band] = p.digest(args.intake/'research_audit'/'tables'/
                                             f'{band}_P6_transmitance__List1.csv')
            domains.append((band,tuple(str(v) for v in waves),
                            np.array([values[p.key(r)] for r in rows]),flags))
        config['hashes']['source_flags'] = p.digest(
            args.intake/'research_audit'/'uvis_consistency_issues.json')
        p.save_json(destination/'provenance_indices.json',config)
        results = []
        for domain,targets,y,flags in domains:
            for sensitivity in ('source_retained','source_flag_sensitivity') if flags else ('source_retained',):
                keep = np.isfinite(y).all(axis=1)
                if sensitivity=='source_flag_sensitivity':
                    keep &= np.array([p.key(r) not in flags for r in rows])
                records = [r for i,r in enumerate(rows) if keep[i]]
                yy = y[keep]
                for mode in ('genotype','treatment'):
                    identity = f'indices__paired_primary__{domain}__{sensitivity}__{mode}'
                    signature = hashlib.sha256(json.dumps(dict(config=config,
                        identity=identity,keys=[p.key(r) for r in records]),sort_keys=True).encode()).hexdigest()
                    result = p.run_experiment(destination/identity,records,yy,targets,mode=mode,
                        cohort='paired_primary',domain=domain,image=True,flags=flags,
                        signature=signature,nulls=True,designs=designs(records,mode),
                        null_models_override=['metadata_plus_indices'])
                    results.append(result)
                    p.save_json(state,dict(status='running',completed_experiments=len(results),
                                           last_experiment=identity))
        p.save_csv(destination/'metrics_indices.csv',[s for result in results for s in result['summary']])
        p.save_json(destination/'summary_indices.json',dict(status='completed',config=config,
                                                           experiments=len(results),results=results))
        subprocess.run([sys.executable,str(Path(__file__).with_name('prediction_audit.py')),
                        '--intake',str(args.intake),'--run',str(args.run),'--repo',str(args.repo)],
                       check=True)
        p.save_json(state,dict(status='completed',experiments=len(results)))
    except Exception as exc:
        p.save_json(state,dict(status='failed',error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
