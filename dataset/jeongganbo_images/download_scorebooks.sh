AJAENG="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%EC%95%84%EC%9F%81%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11354"
DAEGEUM="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%EB%8C%80%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11356"
GAYAGEUM="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%EA%B0%80%EC%95%BC%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11352"
GEOMUNGO="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%EA%B1%B0%EB%AC%B8%EA%B3%A0%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11351"
HAEGEUM="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/17&filename=2021%20%ED%95%B4%EA%B8%88%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11353"
PIRI="https://www.gugak.go.kr/site/inc/file/fileDownload?dirname=/board/15&filename=2021%20%ED%94%BC%EB%A6%AC%EC%A0%95%EC%95%85%EB%B3%B4.pdf&type=F&boardid=11350"
mkdir ./scorebooks
cd ./scorebooks
echo "DOWNLOADING AJAENG SCOREBOOK..."
curl $AJAENG --output ./ajaeng.pdf
echo "DOWNLOADING DAEGEUM SCOREBOOK..."
curl $DAEGEUM --output ./daegeum.pdf
echo "DOWNLOADING GAYAGEUM SCOREBOOK..."
curl $GAYAGEUM --output ./gayageum.pdf
echo "DOWNLOADING GEOMUNGO SCOREBOOK..."
curl $GEOMUNGO --output ./geomungo.pdf
echo "DOWNLOADING HAEGEUM SCOREBOOK..."
curl $HAEGEUM --output ./haegeum.pdf  
echo "DOWNLOADING PIRI SCOREBOOK..."
curl $PIRI --output ./piri.pdf