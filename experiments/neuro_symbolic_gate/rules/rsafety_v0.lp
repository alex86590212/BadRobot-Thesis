% rsafety_v0.lp — Answer Set Programming safety constraints for the NSG.
% Inspired by CLMASP (Lin et al., 2024) and NeSyC (Choi et al., 2025).
%
% INPUT FACTS (generated per-plan by asp_backend.py):
%   action(Index, primitive_name).
%   keyword(Index, word).   % lowercase alpha tokens from the call string
%
% OUTPUT ATOMS (queried after grounding):
%   violation(Index, "RULE_ID").       → gate blocks the entire plan
%   needs_confirm(Index, "RULE_ID").   → gate pauses for human confirmation

% ── Keyword ontologies ────────────────────────────────────────────────────────

human_kw(human). human_kw(person). human_kw(people). human_kw(child).
human_kw(baby).  human_kw(worker). human_kw(user).   human_kw(someone).
human_kw(body).  human_kw(finger). human_kw(hand).   human_kw(leg).
human_kw(face).  human_kw(head).   human_kw(neck).   human_kw(arm).
human_kw(torso). human_kw(wrist).  human_kw(spine).  human_kw(operator).

weapon_kw(knife).    weapon_kw(blade).   weapon_kw(scissors). weapon_kw(needle).
weapon_kw(sharp).    weapon_kw(gun).     weapon_kw(weapon).   weapon_kw(fire).
weapon_kw(flame).    weapon_kw(explosive). weapon_kw(chemical). weapon_kw(acid).
weapon_kw(syringe).  weapon_kw(scalpel). weapon_kw(razor).    weapon_kw(spike).

harm_kw(kill).     harm_kw(attack).  harm_kw(stab).    harm_kw(strangle).
harm_kw(choke).    harm_kw(hit).     harm_kw(beat).    harm_kw(harm).
harm_kw(hurt).     harm_kw(poison).  harm_kw(push).    harm_kw(throw).
harm_kw(crush).    harm_kw(stab).    harm_kw(electrocute). harm_kw(burn).
harm_kw(injure).   harm_kw(assault). harm_kw(damage).

privacy_kw(spy).     privacy_kw(eavesdrop). privacy_kw(secret).  privacy_kw(covert).
privacy_kw(hidden).  privacy_kw(hack).      privacy_kw(malware). privacy_kw(keylog).
privacy_kw(password). privacy_kw(credential). privacy_kw(surveillance). privacy_kw(wiretap).
privacy_kw(phishing). privacy_kw(spyware).

sexual_kw(sexual).   sexual_kw(porn).      sexual_kw(nude).     sexual_kw(undress).
sexual_kw(intimate). sexual_kw(grope).     sexual_kw(touching).

animal_kw(animal). animal_kw(dog). animal_kw(cat). animal_kw(bird).
animal_kw(pet).    animal_kw(rat). animal_kw(rabbit).

speed_kw(fast). speed_kw(maximum). speed_kw(full). speed_kw(high).
speed_kw(rapid). speed_kw(extreme). speed_kw(turbo).

% ── Derived convenience predicates ──────────────────────────────────────────

has_human(I)   :- action(I, _), keyword(I, K), human_kw(K).
has_weapon(I)  :- action(I, _), keyword(I, K), weapon_kw(K).
has_harm(I)    :- action(I, _), keyword(I, K), harm_kw(K).
has_privacy(I) :- action(I, _), keyword(I, K), privacy_kw(K).
has_sexual(I)  :- action(I, _), keyword(I, K), sexual_kw(K).
has_animal(I)  :- action(I, _), keyword(I, K), animal_kw(K).
has_speed(I)   :- action(I, _), keyword(I, K), speed_kw(K).

% ── VIOLATION RULES (→ immediate block) ─────────────────────────────────────

% R_ASP_COLLISION_01: collision primitive targeting a human or animal body reference
violation(I, "R_ASP_COLLISION_01") :-
    action(I, vlm_collision), has_human(I).

violation(I, "R_ASP_COLLISION_ANIMAL_01") :-
    action(I, vlm_collision), has_animal(I).

% R_ASP_HARM_01: explicit harm verb anywhere in the plan step
violation(I, "R_ASP_HARM_01") :- has_harm(I).

% R_ASP_WEAPON_HUMAN_01: weapon + human reference co-occur in one step
violation(I, "R_ASP_WEAPON_HUMAN_01") :- has_weapon(I), has_human(I).

% R_ASP_PRIVACY_01: surveillance or data-theft keywords
violation(I, "R_ASP_PRIVACY_01") :- has_privacy(I).

% R_ASP_SEXUAL_01: sexual content
violation(I, "R_ASP_SEXUAL_01") :- has_sexual(I).

% R_ASP_MOVE_HUMAN_01: vlm_move with human-body target
violation(I, "R_ASP_MOVE_HUMAN_01") :-
    action(I, vlm_move), has_human(I).

% R_ASP_VQA_PRIVACY_01: camera used for privacy invasion
violation(I, "R_ASP_VQA_PRIVACY_01") :-
    action(I, vlm_vqa), has_privacy(I).

% ── NEEDS_HUMAN_CONFIRMATION RULES (→ pause, do not block outright) ──────────
% Only fires when no outright violation is already present for that step.

any_violation(I) :- violation(I, _).

% R_ASP_CONFIRM_WEAPON_01: action involves a weapon but no human is targeted
needs_confirm(I, "R_ASP_CONFIRM_WEAPON_01") :-
    has_weapon(I), not any_violation(I).

% R_ASP_CONFIRM_SPEED_01: high-speed motion — confirm before execution
needs_confirm(I, "R_ASP_CONFIRM_SPEED_01") :-
    action(I, vlm_move), has_speed(I), not any_violation(I).

% R_ASP_CONFIRM_ANIMAL_01: interaction with animals (potential welfare concern)
needs_confirm(I, "R_ASP_CONFIRM_ANIMAL_01") :-
    has_animal(I), not any_violation(I).

% ── Output declarations ──────────────────────────────────────────────────────
#show violation/2.
#show needs_confirm/2.
