from pathlib import Path
import subprocess


PDF_URL = dict(
  ajaeng="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%EC%95%84%EC%9F%81%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11354", 
  daegeum="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%EB%8C%80%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11356", 
  gayageum="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%EA%B0%80%EC%95%BC%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11352", 
  geomungo="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%EA%B1%B0%EB%AC%B8%EA%B3%A0%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11351", 
  haegeum="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%ED%95%B4%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11353", 
  piri="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%ED%94%BC%EB%A6%AC%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11350", 
)


def download_scorebooks(jeongganbo_dir:Path):  
  scorebook_dir = jeongganbo_dir / 'scorebooks'
  scorebook_dir.mkdir(parents=True, exist_ok=True)
  
  for inst, url in PDF_URL.items():
    save_path = scorebook_dir / f'{inst}.pdf'
    
    if save_path.exists():
      print(f"{inst} scorebook already exists. Skipping download.")
      continue
    
    print(f"Downloading {inst} scorebook...")
    try:
      subprocess.run(['curl', url, '-o', str(save_path)], check=True)
    except subprocess.CalledProcessError as e:
      print(f"Failed to download {inst} scorebook: {e}")
      continue