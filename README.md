# ASR Benchmarking Tool 🎙️

เครื่องมือสำหรับทดสอบประสิทธิภาพ (Benchmarking) ของโมเดล Automatic Speech Recognition (ASR) ที่รองรับภาษาไทยและภาษาอังกฤษ โดยเน้นการวัดผลที่ครอบคลุมทั้งด้านความแม่นยำและความเร็วในการประมวลผล

## 🌟 คุณสมบัติหลัก

- **รองรับโมเดลหลากหลายสถาปัตยกรรม:**
  - **ESPnet Conformer (BLSTM Baseline):** โมเดลมาตรฐานสำหรับการเปรียบเทียบ
  - **NeMo Fast-Conformer:** โมเดลประสิทธิภาพสูงจาก NVIDIA
  - **Squeezeformer:** โมเดลที่เน้นความเร็วและประหยัดทรัพยากร
  - **Typhoon ASR:** โมเดล ASR ล่าสุดที่รองรับการทำงานแบบ Real-time
- **การวัดผลที่ละเอียด (Metrics):**
  - **WER (Word Error Rate):** วัดความแม่นยำของการถอดความ
  - **Latency:** เวลาเฉลี่ยที่ใช้ในการประมวลผลต่อหนึ่งประโยค
  - **RTF (Real-time Factor):** อัตราส่วนเวลาประมวลผลต่อความยาวเสียง (ยิ่งน้อยยิ่งเร็ว)
  - **GMACs:** ค่าความซับซ้อนของการคำนวณ (ต่อเสียง 1 วินาที)
  - **Parameter Count:** จำนวนพารามิเตอร์ของโมเดล
- **Dataset:** ใช้ข้อมูลจาก `typhoon-ai/TVSpeech` (Hugging Face)
- **Visualization:** มีเครื่องมือสร้างกราฟเปรียบเทียบผลลัพธ์โดยอัตโนมัติ

## 📂 โครงสร้างโปรเจกต์

```text
.
├── main.py              # ไฟล์หลักสำหรับรัน Benchmark ทั้งหมด
├── DrawGraphs.py        # เครื่องมือสร้างกราฟจากผลลัพธ์ (CSV)
├── datasets/            # จัดการการโหลดและประมวลผลข้อมูล
├── models/              # Wrapper สำหรับโหลดโมเดลแต่ละประเภท
├── utils/               # เครื่องมือเสริม เช่น Logger และ System Check
├── logs/                # เก็บประวัติการทำงาน (Runtime Logs)
└── asr_benchmark_results.csv # ไฟล์สรุปผลลัพธ์หลังรันเสร็จ
```

## 🚀 การติดตั้ง

โปรเจกต์นี้จัดการ Dependencies ด้วย `uv` (แนะนำ) หรือสามารถใช้ `pip` ปกติได้

### แบบที่ 1: ใช้ `uv` (แนะนำ)
```bash
# ติดตั้ง dependencies ทั้งหมดตาม lockfile
uv sync
```

### แบบที่ 2: ใช้ `pip`
```bash
pip install torch librosa numpy pandas soundfile datasets jiwer ptflops fvcore matplotlib
pip install nemo_toolkit[asr] espnet_model_zoo
```

## 🛠️ วิธีการใช้งาน

### 1. การรัน Benchmark
รันไฟล์ `main.py` เพื่อเริ่มการทดสอบโมเดลทั้งหมดที่กำหนดไว้ในสคริปต์:
```bash
python main.py
```
*ระบบจะโหลด Dataset จาก Hugging Face อัตโนมัติ และบันทึกผลลัพธ์ลงใน `asr_benchmark_results.csv`*

### 2. การสร้างกราฟเปรียบเทียบ
หลังจากรัน Benchmark เสร็จแล้ว คุณสามารถสร้างกราฟเพื่อดูความแตกต่างได้:
```bash
python DrawGraphs.py
```
ระบบจะแสดงเมนูให้เลือก Metric ที่ต้องการวาดกราฟ (เช่น WER, RTF, GMACs) และเลือกบันทึกรูปภาพได้

## 📊 ผลลัพธ์และการบันทึกข้อมูล
- **CSV Output:** ผลลัพธ์จะถูกบันทึกในรูปแบบตารางที่อ่านง่ายในไฟล์ `asr_benchmark_results.csv`
- **Logging:** รายละเอียดการทำงานและ Error ต่างๆ จะถูกบันทึกไว้ในโฟลเดอร์ `logs/` แบ่งตามช่วงเวลาที่รัน

## ⚖️ License
โปรเจกต์นี้เผยแพร่ภายใต้ [MIT License](LICENSE)
