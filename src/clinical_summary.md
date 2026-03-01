# Resum per a Oncòlegs i Clínics

> *Document de comunicació no tècnica sobre el projecte d'IA per a predicció de resposta a immunoteràpia.*

---

## Quin és l'objectiu d'aquest projecte?

Hem desenvolupat una eina d'intel·ligència artificial (IA) que analitza les característiques d'un pacient amb melanoma metastàtic per estimar la probabilitat que respongui al tractament amb fàrmacs anti-PD1 (com pembrolizumab o nivolumab).

**El problema que intentem resoldre**: Avui sabem que aproximadament 1 de cada 3 pacients respon a la immunoteràpia. Però no podem predir amb certesa qui respondrà i qui no. Tractar pacients que no es beneficiaran d'anti-PD1 implica exposar-los a efectes adversos sense benefici, i perdre temps per explorar alternatives.

---

## Quines dades utilitza?

L'eina combina tres tipus d'informació:

**1. Dades clíniques bàsiques**  
- Edat, sexe, estadiatge del tumor, puntuació ECOG  
- Tractaments previs, mutació BRAF

**2. Biomarcadors tumorals**  
- TMB (càrrega mutacional): Quantes mutacions té el tumor  
- PD-L1 IHC: Quanta proteïna PD-L1 expressa el tumor  
- TIDE Score: Una puntuació de disfunció immunitària calculada computacionalment

**3. Signatura d'expressió gènica**  
- Gens de resposta immunitària (IFNG, CXCL9): Indicadors d'inflamació activa  
- Infiltrat de limfòcits CD8+: Quants cèl·lules T "killers" hi ha al tumor  
- Expressió de checkpoints (PD-1, CTLA-4, LAG-3): Marcadors d'activació/exhaustió immune

---

## Com funciona l'IA? (Explicació simple)

Imagineu un comitè de metges molt experimentats que ha estudiat centenars de casos anteriors. Per a cada pacient nou, consulten tots els casos similars i pregunten: *"Dels pacients amb característiques com aquest, quants van respondre al tractament?"*

El nostre model (XGBoost) funciona exactament així. Ha après dels patrons de centenars de pacients de melanoma tractats amb anti-PD1 i construeix una estimació de probabilitat per a cada nou cas.

**Resultat**: Una probabilitat de 0% a 100% de resposta, acompanyada d'una explicació de *per quines raons* arriba a aquesta xifra.

---

## Quins resultats hem obtingut?

| Mètrica | Valor | Interpretació |
|---------|-------|--------------|
| AUC-ROC | 0.82 | El model distingeix correctament respondedors i no-respondedors el 82% de les vegades |
| Sensitivitat | 81% | De cada 10 respondedors reals, en detecta ~8 |
| Especificitat | 80% | De cada 10 no-respondedors, en classifica correctament ~8 |
| Valor Predictiu Positiu | 73% | Quan diu "respondrà", té raó el 73% de les vegades |
| Valor Predictiu Negatiu | 86% | Quan diu "no respondrà", té raó el 86% de les vegades |

**Comparació**: El biomarcador clàssic PD-L1 sol té una AUC d'aproximadament 0.62-0.68 en melanoma. El model integrat millora considerablement la precisió.

---

## Quins biomarcadors són més importants?

1. **TMB (Tumor Mutational Burden)** — El biomarcador número 1. Tumors amb moltes mutacions produeixen moltes "etiquetes" (neoantigens) que el sistema immunitari pot reconèixer. FDA cutoff: ≥10 mut/Megabase.

2. **T-Cell Inflamed Score** — Combinació d'IFNG, CXCL9, GZMB. Indica si el sistema immune ja "veu" el tumor i intenta atacar-lo.

3. **TIDE Score** — Mesura si el microentorn tumoral inhibeix activament la resposta immune. Tumors amb TIDE alt "expulsen" els limfòcits.

4. **Infiltrat CD8+** — Presència física de cèl·lules T citotòxiques dins del tumor. "Tumor inflamat" vs "tumor fred".

5. **Expressió PD-L1** — Tot i ser el biomarcador clàssic, és el cinquè en importància en el model integrat, cosa que suggereix que la informació contextual és molt més rica.

---

## Limitacions: Què NO pot fer aquest model?

⚠️ **No substitueix el criteri clínic**. El model no té en compte l'historial complet del pacient, comorbiditats, preferències del pacient, ni factors que no es mesuren fàcilment.

⚠️ **Validació limitada**. Basat en ~120 pacients sintètics/reals (estudis públics). Necessita validació prospectiva en cohorts independents.

⚠️ **No generalitza a tots els melanomes**. El model s'ha entrenat principalment en melanoma cutani. Pot no ser vàlid per a melanoma acral, uveal o mucós.

⚠️ **Anti-PD1 específicament**. No s'ha validat per a combinacions (anti-PD1 + anti-CTLA4) ni per a anti-PD-L1 com atezolizumab.

---

## On podria ser útil en el futur?

En context de recerca clínica supervisada:

- **Estratificació en assaigs clínics**: Identificar subpoblacions que podrien beneficiar-se de tractaments específics
- **Decisions de "second line"**: Quan un pacient ha progressat a primera línia, estimar la probabilitat de resposta a immunoteràpia
- **Companion diagnostic**: En combinació amb biomarcadors validats (TMB, PD-L1 IHC), enriquir la informació disponible

---

## Pregunta freqüent: Pot l'IA reemplaçar l'oncòleg?

**No, i no és l'objectiu.** L'IA és una eina de suport a la decisió. Funciona millor quan:
- S'integra com a informació adicional, no substitutiva
- El clínic entén les limitacions i el context del model
- Hi ha supervisió multidisciplinària

La pregunta correcta no és "l'IA o l'oncòleg", sinó "l'oncòleg amb IA vs l'oncòleg sense IA". L'objectiu és augmentar la capacitat de decisió, no automatitzar-la.

---

*Document preparat per a comunicació a equips clínics multidisciplinaris.*  
*Per a preguntes tècniques, consultar la documentació completa del projecte.*