# Stima Monoculare della Profondità con cGAN
Questo progetto esplora la stima supervisionata della profondità monoculare da immagini RGB utilizzando le **Reti Avversarie Generative Condizionate (cGAN)**. L'obiettivo è generare mappe di profondità accurate attraverso l'apprendimento avversario, addestrando sia un **generatore** che un **discriminatore** su dataset RGB-D.

## Panoramica
- **Problema**: La stima della profondità da una singola immagine è un compito cruciale in ambiti come la robotica, la realtà aumentata e la modellazione 3D.
- **Approccio**: Utilizzo delle cGAN per generare mappe di profondità di alta qualità.
  - **Generatore**: Architettura U-Net per una stima grezza della profondità, affinata da una rete di refinement.
  - **Discriminatore**: Rete convoluzionale per distinguere mappe di profondità reali da quelle generate.

## Caratteristiche Principali
- **U-Net come Generatore**: Cattura efficacemente le informazioni posizionali con connessioni skip.
- **Processo di Addestramento**:
  1. Addestramento della U-Net (Global Net) con una funzione di perdita L1.
  2. Pre-addestramento della Refinement Net per ottimizzare i residui.
  3. Addestramento avversario tra Refinement Net e Discriminatore.
- **Funzioni di Perdita**:
  - Perdita di ricostruzione (L1).
  - Perdita avversaria (Binary Cross-Entropy).
- **Ottimizzazione**: Uso misto di ottimizzatori SGD e Adam per i vari componenti della rete.
  
## Lavori Futuri
- Esplorare meccanismi di scheduling del learning rate per migliorare la stabilità dell'addestramento.
- Sperimentare con immagini di dimensioni maggiori per una migliore generalizzazione.

## Riferimenti
Principali riferimenti utilizzati nel progetto:
1. [Panoramica sulla Stima Monoculare della Profondità](https://dx.doi.org/10.1007/s11431-020-1582-8)
2. [Reti Avversarie Condizionate per la Predizione della Profondità](https://arxiv.org/abs/1808.07528)
3. [U-Net per la Segmentazione delle Immagini](https://arxiv.org/abs/1505.04597)
