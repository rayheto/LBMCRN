# LBMCRN v1.0 model card

LBMCRN is a 29,220-parameter convolutional recurrent model for mono 16 kHz speech enhancement. It uses a causal 512/256 STFT, two low-rank band-mixing and temporal GRU blocks, and a complex ratio mask.

The selected weights are the seed-43 parent checkpoint from the development comparison. The pretrained file contains only the model state dictionary. Its source checkpoint identity and exported file SHA-256 are recorded in checkpoints/manifest.json.

## Development quality

Mean scores across 1,792 paired utterances are STOI 0.82165, ESTOI 0.69227, WB-PESQ 1.97511, SI-SDR 13.20544 dB and SI-SDR improvement 5.19449 dB.

The DNS5 subset with measured input SNR below 6 dB contains 420 utterances. It scores STOI 0.70210, ESTOI 0.50279, WB-PESQ 1.41280 and SI-SDR 6.35562 dB. Low-SNR intelligibility remains below NSNet2.

The README and interactive quality table provide matched NSNet2, NSNet3 and official GTCRN DNS3 comparisons. The development inputs were repeatedly used during model selection. Results do not constitute independent generalization or human listening acceptance.

## Runtime

The Python entry processes complete files while retaining causal network history across each file. A bounded-memory frame API remains pending validation.

The selected v1.0 weights have been exported to INT16 and measured on XIAO ESP32-S3. Three paced runs measured 9.072 ms median processing time and 56.82% CPU0 with GRU working arrays in PSRAM. A 10,000-frame run had zero frames exceeding 16 ms. All 16,512 mask values across 64 frames exactly matched the integer reference. Full PCM quality and I2S acceptance remain pending.
