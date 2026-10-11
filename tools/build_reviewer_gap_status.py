#!/usr/bin/env python3
"""Build an evidence-indexed status of the review objections."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text())
def main():
    strict=read('scores/p1_strict_fresh_adaptive_vs_parser_fixed1024_noclamp_20261002.json')
    c512=read('scores/p1_strict_fresh_adaptive_vs_fixed512_noclamp_20261002.json')
    full=read('scores/p1_strict_fresh_adaptive_vs_fullhistory_noclamp_20261002.json')
    operational=read('scores/operational_forecasting_verified_20261003.json')
    guard=read('scores/numeric_claim_guard_fresh_e2e_20261003.json')
    external=read('scores/agentic/external_protocol_audit.json')
    variance=read('scores/run_variance_overlap_20261003.json')
    stochastic=read('scores/sampling_variance_probe_20261005.json')
    stochastic_protocol=read('scores/sampling_variance_protocol_summary_20261006.json')
    alias=read('scores/alias_specialist_completion_20261005.json')
    chronosteer=read('scores/agentic/chronosteer_official_reproduction_20261005.json')
    oracle=read('scores/context_oracle_retrospective_20261003.json')
    normalized=read('results/controller_fresh_full_strict_normalized_20261003/normalization_audit.json')
    status={
      'strict_fresh_adaptive_vs_fixed1024': {'status':'partial','rows':strict['strict_common_rows'],'paired':strict['paired'],'conclusion':'small aggregate MAE/RMSE improvement, but not universal across metrics/dimensions'},
      'strict_fresh_adaptive_vs_fixed512': {'status':'supported','rows':c512['strict_common_rows'],'paired':c512['paired'],'conclusion':'adaptive is substantially better on aggregate error/sMAPE than short fixed context'},
      'strict_fresh_adaptive_vs_full_history': {'status':'not_superior','rows':full['strict_common_rows'],'paired':full['paired'],'conclusion':'no general superiority; relative sMAPE favors full history'},
      'simple_same_language_interface': {'status':'controlled','evidence':'scores/simple_pipeline_equivalence_audit.json','conclusion':'fixed parser and simple policy trajectories identical on 220 common rows'},
      'operational_forecast_only': {'status':'supported_protocol','evidence':'scores/operational_forecasting_verified_20261003.json','verified_target_rows':operational['verified_target_rows'],'all_model_common_rows':operational['all_model_common_rows'],'models':list(operational['all_model_matched'])},
      'numeric_grounding': {'status':'guarded','evidence':'scores/numeric_claim_guard_fresh_e2e_20261003.json','raw_claims':guard['raw_claims'],'raw_unsupported':guard['raw_unsupported'],'post_guard_unresolved':guard['post_guard_unresolved']},
      'repeat_run_variance': {'status':'deterministic_reproducibility_measured','evidence':'scores/run_variance_overlap_20261003.json','pairs':variance['pairs']},
      'stochastic_sampling_variance': {'status':'diagnostic_completed','evidence':'scores/sampling_variance_protocol_summary_20261006.json','primary_probe':stochastic_protocol['primary_probe'],'thinking_enabled_replication':stochastic_protocol['thinking_enabled_replication'],'summary':stochastic['summary'],'temperature':stochastic['temperature'],'top_p':stochastic['top_p']},
      'alias_specialist_completion': {'status':'completed_on_mps','evidence':'scores/alias_specialist_completion_20261005.json','rows_completed':alias['rows_completed'],'scored_rows':alias['scored_rows'],'unscored_rows':alias['unscored_rows']},
      'controller_normalization': {'status':'audited_alias_normalization','evidence':'results/controller_fresh_full_strict_normalized_20261003/normalization_audit.json','before_valid':normalized['before_valid'],'after_valid':normalized['after_valid'],'changed_rows':normalized['changed_rows']},
      'retrospective_fixed_context_oracle': {'status':'upper_bound_only','evidence':'scores/context_oracle_retrospective_20261003.json','common_rows':oracle['common_rows'],'oracle':oracle['oracle']},
      'case_study_target_leakage': {'status':'fixed_in_source','evidence':'scores/case_study_leakage_audit.json'},
      'external_agent_official_reproduction': {'status':'chronosteer_completed; other_agents_not_comparable','evidence':'scores/agentic/external_protocol_audit.json','methods':{k:v.get('status') for k,v in external['methods'].items()},'official_result':'scores/agentic/chronosteer_official_reproduction_20261005.json','chronosteer_summary':chronosteer['results']},
      'remaining_caveats':['Stochastic variance is diagnostically estimated on a completed 10-sample, 3-repeat probe plus a separate 4-sample thinking-enabled pilot; production remains greedy and no stochastic TLS rank is claimed.', 'ChronoSteer checkpoint-level reproduction is complete, but its benchmark and event-context protocol are not aligned with the current manifest, so it is an external reference rather than a paired rank.', 'Legacy full TLS tables should not be interpreted as pure forecasting accuracy; use operational_forecasting_verified for forecast-only claims.']
    }
    out=ROOT/'scores/reviewer_gap_status_20261003.json'; out.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
