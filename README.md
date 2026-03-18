# ASR Benchmarking Tool

โปรเจกต์นี้เป็นเครื่องมือสำหรับทดสอบประสิทธิภาพ (Benchmarking) ของโมเดล Automatic Speech Recognition (ASR) ต่างๆ โดยเน้นที่ภาษาไทยและภาษาอังกฤษ โดยใช้ Dataset จาก `typhoon-ai/TVSpeech`

## คุณสมบัติหลัก

- **รองรับหลายโมเดล:**
  - ESPnet BLSTM (Baseline)
  - NeMo Fast-Conformer
  - Squeezeformer
  - Typhoon ASR (Real-time)
- **การวัดผลที่ครอบคลุม:**
  - **WER (Word Error Rate):** ความแม่นยำของการถอดความ
  - **Latency:** เวลาที่ใช้ในการประมวลผลต่อตัวอย่าง
  - **RTF (Real-time Factor):** อัตราส่วนเวลาประมวลผลต่อความยาวเสียง
  - **GMACs:** ประมาณการความซับซ้อนของการคำนวณ (ต่อเสียง 1 วินาที)
  - **Parameter Count:** จำนวนพารามิเตอร์ของโมเดล
- **ระบบ Logging แบบรวมศูนย์:** บันทึกข้อมูลการรันทั้งหมดลงในไฟล์เดียวต่อการรันหนึ่งครั้งในโฟลเดอร์ `logs/`
- **การบันทึกผลลัพธ์:** ส่งออกผลลัพธ์ในรูปแบบ CSV (`asr_benchmark_results.csv`)

## โครงสร้างโปรเจกต์

- `main.py`: ไฟล์หลักสำหรับการรัน Benchmark
- `models/`: โฟลเดอร์เก็บ Wrapper สำหรับโหลดโมเดลต่างๆ
- `utils/`: เครื่องมือช่วยเหลือ เช่น ระบบ Logger และการเช็ค GPU compatibility
- `logs/`: โฟลเดอร์เก็บไฟล์ Log ของแต่ละการรัน
- `datasets/`: โค้ดที่เกี่ยวข้องกับการจัดการข้อมูล

## การใช้งาน

1. ติดตั้ง Dependencies (แนะนำให้ใช้ `uv` หรือ `pip`):
   ```bash
   pip install torch librosa numpy pandas soundfile datasets jiwer ptflops fvcore nemo_toolkit[asr] espnet_model_zoo
   ```

2. รัน Benchmark:
   ```bash
   python main.py
   ```