# Requirement → test trace

Every rule the design states, and the test that holds it up. This exists
because the reverse order was used once and it cost real time: `met_demands`
was deleted on a call-graph reading, and the `require` gate went with it —
a text-only model could then claim `require: [vision]`, an id promising a
capability the entry does not advertise.

The rule that prevents that: **design → docs → tests → code.** A requirement
is not implemented until a test named here fails without the code. Code is
never removed on a call-graph reading alone.

Run `python extras/check_trace.py` to verify every test named below exists
and every one of them carries a `Validates:` citation.

## The task is a declaration

| # | Requirement | Test |
|---|---|---|
| 1 | The task comes from the sidecar `tasks:` list; nothing infers or overrides it | `test_task_ids_resolve_to_declared_demands` |
| 2 | An unknown task id warns and is ignored — only catalog ids exist | `test_resolve_tasks_ignores_unblessed_tag_with_warning` |
| 3 | Each satisfiable task emits its own id | `test_every_category_gets_an_entry` |
| 4 | A task that cannot be met at any rung emits **nothing** | `test_rung_is_none_when_the_category_cannot_be_served`, `test_plan_vision_and_text_both_over_budget_disables` |
| 5 | Every model also gets a fallback entry with no task segment | `test_untagged_gets_a_bare_id`, `test_every_model_gets_a_fallback` |
| 6 | The fallback can itself fail and emit nothing | `test_model_with_no_rung_at_all_emits_nothing` |

## Vision and the ladder

| # | Requirement | Test |
|---|---|---|
| 7 | Vision is shed when shedding it buys context | `test_rung_drops_vision_when_it_buys_context`, `test_category_prefers_context_over_capability` |
| 8 | Vision is kept when it buys nothing — no capability discarded for no gain | `test_rung_keeps_vision_when_it_is_free` |
| 9 | A rung that does not fit is skipped, and the next along the order tried | `test_rung_honours_keep_then_falls_back_to_vision` |
| 10 | A **category** takes the rung buying the most context | `test_category_prefers_context_over_capability` |
| 11 | The **fallback** takes the first qualifying rung — capability first | `test_fallback_keeps_capability_until_the_minimum_forces_it` |
| 12 | The fallback sheds only when holding a capability leaves no legal serving | `test_fallback_sheds_when_nothing_else_fits` |
| 13 | `sacrifice:` order drives the ladder, not a global order | `test_shed_levels_are_prefixes_of_the_declared_order` |
| 14 | Capabilities a category omits are shed last, in the `shed:` order | `test_unnamed_capabilities_come_last_not_first`, `test_shed_order_is_configurable` |
| 15 | `require:` is never shed | `test_required_capabilities_are_never_shed` |
| 16 | `require:` gates **existence** — a model lacking it emits nothing | `test_require_gates_existence`, `test_require_prevents_the_entry_emitted` |
| 17 | `keep:` forbids shedding but does **not** gate existence | `test_keep_protects_from_shedding_without_gating_existence` |
| 18 | `keep:` is enforced against a real draft axis | `test_keep_draft_is_enforced_now_that_the_axis_exists` |

## Context floors

| # | Requirement | Test |
|---|---|---|
| 19 | `context_length:` is a ceiling, applied in **both** solve branches | `test_sidecar_ceiling_binds_in_the_squeeze_branch`, `test_sidecar_ceiling_is_a_min_not_a_fixed_cap` |
| 20 | Native vision (no mmproj file) still gets the image floor | `test_solve_context_image_floor_native_vision` |
| 21 | Only `--max-context` pins; `context_length:` is never a floor | `test_serving_pin_ignores_the_design_max`, `test_explicit_max_context_still_pins_and_floors`, `test_design_max_is_a_ceiling_not_a_floor` |
| 22 | A declared category outranks the tools heuristic | `test_category_floor_beats_the_tools_heuristic` |
| 23 | `min_context:` and `--max-context` outrank a category | `test_category_floor_never_overrides_the_operator` |

## Image resolution budget

| # | Requirement | Test |
|---|---|---|
| 24 | A megapixel budget converts to tokens through the projector geometry | `test_min_image_megapixels_converts_through_geometry` |
| 25 | The image floor includes the output reserve | `test_min_image_megapixels_floor_includes_output_reserve` |
| 26 | A projector that does not scale with pixels is warned and ignored | `test_min_image_megapixels_ignored_on_non_proportional` |
| 27 | A budget above the declared cap is warned and clamped | `test_min_image_megapixels_too_large_warns` |
| 28 | `min_image_megapixels: 0` drops the requirement and warns | `test_min_image_megapixels_zero_allows_shedding` |
| 29 | A megapixel budget keeps vision from being shed | `test_min_image_megapixels_keeps_vision` |
| 30 | No budget leaves vision shed-able; 0 drops the requirement | `test_no_budget_leaves_vision_shed_able`, `test_zero_budget_drops_the_vision_requirement` |

## Entry ids

| # | Requirement | Test |
|---|---|---|
| 31 | The id carries no context — 182k → 180k must not rename a model | `test_duplicate_ids_do_not_move_when_context_moves` |
| 32 | An untagged entry is the bare stem | `test_untagged_gets_a_bare_id` |
| 33 | A name that cannot be slugged drops the segment, not the entry | `test_unusable_category_name_falls_back_to_a_bare_id` |
| 34 | Ids stay stable across repacks; the disambiguator describes the serving | `test_same_model_two_groups_disambiguated` |

## VRAM safety

| # | Requirement | Test |
|---|---|---|
| 35 | No emitted entry may need more VRAM than exists | `test_solve_context_never_reports_a_context_it_cannot_afford`, `test_overallocation_guard_catches_an_injected_violation` |
| 36 | The guard stays quiet and drops nothing on a sane plan | `test_overallocation_guard_is_quiet_when_the_plan_is_sane` |
| 37 | A CPU-resident model gets no free slots | `test_cpu_resident_chat_gets_no_auto_parallel` (planned) — covered today by `test_solve_context_cpu_resident_returns_design` |
| 38 | `tools` is demoted on the entry's **own** served context | `test_plan_demotes_tools_below_the_threshold_without_a_matrix`, `test_plan_keeps_tools_when_the_context_clears_the_threshold` |

## Allocation

| # | Requirement | Test |
|---|---|---|
| 39 | Residents serve a fixed small context and are charged for it | `test_resident_rag_is_reserved_at_its_fixed_context` |
| 40 | Resident parallelism is ignored — never buys context from chat | `test_resident_parallelism_is_ignored` |
| 43 | The fallback runs the slot solve at exponent 0 | `test_fallback_slot_exponent_is_zero` |
| 44 | Ties in the slot score go to more slots | `test_ties_go_to_more_slots` |
| 87 | A priced resident logs the terms the budget charged (weights + compute + KV at ctx) | `test_selection_reason_names_the_cost_terms`, `test_resident_cost_detail_names_the_terms` |

## Deploy

| # | Requirement | Test |
|---|---|---|
| 45 | Installs are atomic; a truncated tree is never certified | `test_cpp_truncated_dir_is_not_certified_as_installed` |
| 46 | The pruner never removes a bundle a symlink points at | `test_prune_never_removes_the_bundle_a_symlink_points_at` |
| 47 | An ambiguous target selector exits non-zero | `test_ambiguous_prefix_is_an_error_not_a_guess` |
| 48 | `cpu`/`cuda` do not silently install Vulkan | `test_cpu_backend_does_not_install_vulkan` |
| 49 | llama-swap validates as the service user, not root | `test_validate_runs_as_the_service_user` |
| 50 | The unit is verified before it becomes the unit systemd reads | `test_rejected_unit_leaves_the_previous_one_installed`, `test_candidate_name_is_a_valid_unit_name` |
| 51 | An unreadable `HF_HOME` warns instead of crashing the installer | `test_report_resolved_warns_when_hf_home_is_unreadable` |
| 52 | The unit is enableable: it carries `[Install] WantedBy=` | `test_unit_is_enableable` |
| 53 | `InaccessiblePaths=` lists only directories present on this host | `test_render_unit_lists_only_present_hidden_dirs`, `test_inaccessible_paths_omits_absent` |
| 54 | The health probe waits for the socket instead of racing it | `test_probe_waits_for_a_slow_start` |
| 55 | The probe fails, with the journal hint, only after the deadline | `test_probe_gives_up_after_the_deadline` |

## Model listing limits

| # | Requirement | Test |
|---|---|---|
| 56 | `capabilities.context` is the served per-slot window, not the trained maximum | `test_capabilities_context_is_served_window` |
| 57 | The generation ceiling defaults to the served window; a sidecar may lower it, never raise it above the window | `test_sidecar_lowers_max_output_tokens`, `test_max_output_tokens_never_exceeds_the_served_window` |

## Log levels

| # | Requirement | Test |
|---|---|---|
| 58 | A pack run logs at WARNING unless asked: `-v` lifts it to INFO, and only `--dry-run` lifts the floor on its own | `test_pack_run_is_quiet_by_default`, `test_verbose_flag_lifts_the_level`, `test_dry_run_preview_is_visible_without_a_flag`, `test_verbosity_ladder_and_floor` |

## CPU-residency measurement (SPEC.md §Host pool)

| # | Requirement | Test |
|---|---|---|
| 59 | The host parse reads the buffer lines after the projection pass's closing table — the CPU backend's pass-repeated real sizes are never summed twice | `test_parse_host_buffers_reads_final_pass` |
| 60 | The host parse reports host-side component sums beside the device sums; without a projection table the log parses whole | `test_parse_host_buffers_headerless_log_parses_whole` |
| 61 | A `device: cpu` measurement carries the `--n-gpu-layers 0` pin in both the server command and the measurement shape, never duplicated | `test_run_measure_server_cpu_shape_returns_host_buffers`, `test_measurement_shape_stamps_cpu_device` |
| 62 | A CPU-shaped measurement never runs the device-spill rejection; a `-ngl 999` launch that lost its device still measures nothing (spill refuses the host-KV shape) | `test_run_measure_server_cpu_shape_returns_host_buffers`, `test_run_measure_server_gpu_refuses_vulkanless_fullcpu` |
| 63 | `parse_fit_log` falls back to the `Host` report row when no device row exists, so CPU quads derive without new tooling; a device row always wins | `test_parse_fit_log_host_row_fallback`, `test_parse_fit_log_prefers_device_row` |
| 64 | The CPU-shaped affine trio derives the exact quad and persists it under the device-stamped shape | `test_measure_affine_trio_cpu_shape_persists_host_quad` |
| 65 | The device parse stays whole-log: a device run's projection buffers measure 0, host sums never leak into it | `test_parse_device_buffers_whole_log_device_path` |

## Measurement failure evidence

| # | Requirement | Test |
|---|---|---|
| 94 | A failed `llama-fit-params` run records the evidence a later diagnosis needs — argv, model facts, exit code and the full log — in a durable artifact, indexed by one journal row; the warning names the artifact and the cause instead of only the exit code | `test_fit_crash_records_argv_and_stderr`, `test_fit_crash_recorded_once_per_invocation`, `test_fit_timeout_records_argv`, `test_fit_unparseable_records_output` |

## A diagnostic names its subject

| # | Requirement | Test |
|---|---|---|
| 66 | A `tasks:` warning leads with the sidecar's name | `test_usage_warnings_lead_with_the_sidecar_name` |
| 67 | An unhandled-setting warning leads with the sidecar's name | `test_warn_unhandled_names_the_sidecar` |
| 68 | An invalid `allow_profiles` regex warning leads with the sidecar's name | `test_allow_profiles_bad_regex_names_the_sidecar` |
| 69 | A failed `base *` expression warning names the layer it came from | `test_merge_bad_base_expression_names_the_origin` |

## Host budget (SPEC.md §Host pool)

| # | Requirement | Test |
|---|---|---|
| 70 | Host pool floor: an infeasible read clamps the pool to 0, and an over-pool `device: cpu` serving — a variant or a declared fixed category — emits no id | `test_host_pool_clamps_to_zero`, `test_cpu_variant_refused_on_small_pool`, `test_host_unit_refusal_drops_a_category_that_cannot_fit_alone` |
| 71 | Pressure zone: sysctl `(zone high sum, min_free_kbytes floor)` primary; missing/zeroed zoneinfo falls back to 25% of total, called out in the log | `test_pressure_zone_from_zoneinfo`, `test_pressure_zone_fallback_is_logged`, `test_pressure_zone_override_asserted` |
| 72 | `host_pool = min(sys_free, cgroup_free)` with every override arm (`system_ram`, `host_pool` abs/frac) honored | `test_host_pool_mixed_terms`, `test_host_pool_overrides_arm` |
| 73 | Service cgroup caps the pool; no llama-swap cgroup skips the term with a warning | `test_host_pool_mixed_terms`, `test_cgroup_missing_service_skips_term` |
| 74 | A `device: cpu` resident is charged to the host ledger (RAG pair and declared fixed categories); its VRAM charge stays 0 | `test_cpu_resident_charges_host_not_vram`, `test_host_resident_sizing_lifts_cpu_gate`, `test_cpu_fixed_category_resident_charged_host`, `test_host_measurement_does_not_leak_into_vram_pricing` |
| 75 | Host charge is the serving quad at the served context — declared parallel multiplies D and the KV pool and can flip the refusal | `test_cpu_variant_refused_on_small_pool`, `test_cpu_parallel_charge_flips_refusal` |
| 76 | No pool knowledge, no gate — a planner without a host pool never refuses on host grounds | `test_no_host_pool_never_gates` |
| 77 | Every entry carries `metadata.packing`; `device: cpu` entries name device/ctx/parallel/ram and carry no `vram_mib` | `test_metadata_packing_cpu_entry`, `test_metadata_packing_gpu_entry` |
| 78 | `device: cpu` wins the resident role; inside that set the cheapest **measured serving at the role's ctx** wins (file size only when unmeasurable); a selector overrides and is priced; every arm is logged | `test_select_model_prefers_cpu_candidate`, `test_select_model_smallest_among_cpu_candidates`, `test_select_model_no_cpu_keeps_size_rule`, `test_select_model_selector_overrides_cpu_preference`, `test_selection_reason_names_the_arm`, `test_selection_reason_measured_and_selector_arms`, `test_select_model_picks_measured_cheapest_over_file_size`, `test_select_model_falls_back_to_size_when_unmeasurable`, `test_select_model_cpu_preference_beats_cheaper_device`, `test_select_model_selector_prices_only_the_pick` |
| 79 | `profiles.yaml device_roles:` places every model of a named role on the host (logged with its count); absent key is a no-op; unmatched or unknown roles warn; a mapping value other than `cpu` (the key is a role, the value a device) places nothing and warns | `test_device_roles_place_every_role_model`, `test_device_roles_absent_and_unmatched`, `test_device_roles_rejects_unknown_role`, `test_device_roles_rejects_a_non_cpu_value`, `test_device_roles_rejects_a_scalar` |
| 80 | TTL resolves sidecar > task > placement default: `0` disables idle unload, the smallest declared lifetime wins, an undeclared CPU-resident serving gets 3600 s (chat is not special), and an undeclared device entry carries none | `test_ttl_sidecar_wins_when_tighter`, `test_ttl_task_wins_when_tighter`, `test_ttl_zero_means_never_unload_and_wins`, `test_ttl_cpu_helper_defaults_to_host_lifetime`, `test_ttl_cpu_serving_gets_cpu_default`, `test_cpu_chat_entry_gets_cpu_default_ttl`, `test_gpu_entry_has_no_ttl` |
| 81 | A category selector defines the RAG pair too — solve, charge, `emb`/`rnk` vars and resident entry name one model | `test_detect_matrix_category_selector_defines_the_pair` |

## Alien keys

| # | Requirement | Test |
|---|---|---|
| 88 | `role:` is a valid sidecar key; `usage:` warns as alien (the warning points at `tasks:`), and `device:` is `auto`, `cpu`, or a non-negative GPU index | `test_role_is_a_valid_sidecar_key`, `test_usage_is_an_alien_sidecar_key`, `test_unknown_device_value_warns` |
| 89 | An unknown top-level profiles.yaml key warns with a near-miss hint instead of being ignored | `test_unknown_profiles_key_warns_with_a_hint` |
| 106 | `--check` validates profiles.yaml without discovery or a server and names every known error: unknown top-level and nested keys with a near-miss hint, required sections, retired keys, bad types, non-roles and a missing `when:` | `test_check_profiles_names_every_nested_error`, `test_check_profiles_accepts_the_example` |
| 107 | A role is operator data: a label, the trees it owns, and an allocation policy. It gets its own capacity pool; models declare `role:` and tasks declare `device`/`max_context`. A role has no behaviour | `test_custom_role_is_data_and_gets_its_own_pool`, `test_custom_role_parse_reports_every_problem`, `test_custom_role_gets_its_own_size_one_pool`, `test_custom_role_entry_charges_its_own_pool`, `test_task_declares_device_and_max_context` |
| 108 | `pools.ram.spare` keeps that much host RAM out of both the generated `ram` pool and the host ledger | `test_pools_ram_spare_shrinks_the_host_pool` |
| 109 | `matrix.sets` is a mapping of name to expression string; its presence selects the matrix router, its absence capacity; `routing` is retired | `test_check_profiles_validates_sets_and_retired_routing`, `test_matrix_sets_are_accepted_and_select_the_router`, `test_matrix_sets_expand_roles_to_entry_ids` |
| 110 | A role's `share` is a percentage of slack; `share` and `fill` are mutually exclusive, at most one role fills, and shares sum to at most 100% | `test_share_is_a_percentage_of_slack`, `test_share_and_fill_are_validated` |

## Roles (docs/architecture.md §Models, tasks, engines and endpoints)

| # | Requirement | Test |
|---|---|---|
| 90 | The role registry is one table; every engine endpoint is closed and at least one engine serves it (no orphan endpoint) | `test_role_registry_derives_its_sets`, `test_every_engine_endpoint_is_closed_and_served`, `test_infer_backend_gguf_image_goes_to_sd_server` |

## Engine declaration (SPEC.md §Backend Selection)

| # | Requirement | Test |
|---|---|---|
| 95 | A sidecar's `engines:` list is the compatibility claim: bare engine names in preference order, resolved to the first one that is enabled, available, can load the format and serves the task's endpoint; `engines:` wins over `backend:`, nothing resolvable refuses by name, and an absent key keeps inference | `test_engines_vocabulary_is_bare_engine_names`, `test_engines_resolution_honours_declared_order`, `test_engines_resolution_picks_the_transport`, `test_engines_refusal_names_every_engine`, `test_engines_finalize_sets_the_backend`, `test_engines_finalize_refuses_by_name`, `test_engines_wins_over_the_alien_backend_key`, `test_engines_unknown_name_warns_with_a_hint` |

| 96 | An authored sidecar must declare `engines:` (missing refuses, naming the sidecar); `backend:` warns as an alien key and never selects an engine; a model with no authored sidecar keeps inference | `test_sidecar_without_engines_is_refused`, `test_orphan_without_engines_still_infers`, `test_engines_wins_over_the_alien_backend_key`, `test_apply_overrides_backend_is_alien` |

## Endpoint registry (SPEC.md §Backend support matrix)

| # | Requirement | Test |
|---|---|---|
| 97 | One registry owns the closed endpoint set, the task→endpoint map and the engine→endpoints table; engines read their list from it, a bad sidecar value is reported, and the docs that state endpoints are checked against the table | `test_registry_invariants_hold`, `test_registry_task_mapping_is_total`, `test_registry_engine_table_matches_backends`, `test_registry_docs_agree`, `test_registry_unknown_endpoint_is_rejected`, `test_registry_engines_value_shape` |

## Task catalogue (SPEC.md §Tasks)

| # | Requirement | Test |
|---|---|---|
| 98 | The catalogue is the only task vocabulary: `tasks:` ids come from it, a task's `role:` is retired (the model declares its role), and an `audio_cpp.task` cannot contradict the model's task | `test_task_ids_resolve_to_declared_demands`, `test_resolve_tasks_ignores_unblessed_tag_with_warning`, `test_usage_is_an_alien_sidecar_key`, `test_audio_cpp_task_must_match_the_model_task` |

## Declared task vocabulary (SPEC.md §Tasks)

| # | Requirement | Test |
|---|---|---|
| 99 | A task id is declared data: an arbitrary id declared in `profiles.yaml tasks:` serves, and the same id undeclared is not a task and produces no task entry | `test_a_declared_task_id_is_arbitrary`, `test_an_undeclared_task_id_fails` |

## Per-category context cap (SPEC.md §VRAM Allocation)

| # | Requirement | Test |
|---|---|---|
| 92 | A per-category `ctx:` caps the served context at one seam (`resident_ctx`) and is never a floor | `test_category_ctx_caps_at_one_seam` |

## Capacity routing (SPEC.md §Capacity routing)

| # | Requirement | Test |
|---|---|---|
| 100 | The emitted pools are the packer's available VRAM (`default`), the detected host pool (`ram`), and one size-1 pool per scheduling role | `test_capacity_pools_are_gpu_budget_host_pool_and_role_singletons` |
| 101 | Every entry declares the pools it charges, carrying the solve's measurement margin; an unmeasured serving reserves the whole pool and runs alone | `test_capacity_fields_charge_gpu_ram_and_role`, `test_capacity_unestimated_serving_reserves_the_whole_pool` |
| 102 | A `device: cpu` serving charges `ram` only, never a GPU pool | `test_capacity_fields_cpu_helper_charges_host_only` |
| 103 | `evict_cost` prices reload time plus lost prefill for chat only; a sidecar `pp_tps:` overrides the fleet rate | `test_capacity_evict_cost_ignores_prefill_for_stateless_roles`, `test_capacity_evict_cost_uses_sidecar_prompt_rate` |
| 104 | Matrix measurement runs the sibling `llama-fit-params` binary; `llama-server` rejects `--fit-print` | `test_matrix_resolution_measures_with_fit_binary` |
| 105 | A declared GPU fixed category is charged to the VRAM ledger at the emission margin, so the per-model chat solve leaves it room instead of filling the pool | `test_gpu_fixed_category_charges_vram_ledger` |
