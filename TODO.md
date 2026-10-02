# TODO — LLM Quant Bench

## Nouveaux benchmarks inspect_evals

- [x] race_h → abandonné (saturé à 89% sur Ornith 9B)
- [x] MUSR 
- [x] BFCL 

## Scripts / run.sh


### Trous de couverture (non testés)
- Dialogue conversationnel (fluidité naturelle)
- Traduction / multilingue
- Résumé / synthèse de texte long
- Créativité / rédaction

## Nouveaux modèles à benchmarker

### Priorité haute

## Étude KV cache quantisé

- [ ] Qwen3.8-27B (medium) — une fois le rerun terminé (UD-IQ1_S excepté, runs
      systématiquement annulés). Sélectionner un sous-ensemble de quants (le
      plus petit exploitable, un intermédiaire, un Q4, le plus gros) et faire
      varier la quantisation K/V (actuellement q4_0/q4_0 fixe, voir README)
      plutôt que de refaire tourner toute la plage.

## Reporté

- [ ] GPQA — très long à exécuter, retiré du site pour le moment (entrée commentée dans site/models.yaml)