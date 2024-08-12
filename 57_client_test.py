import cv2
import mediapipe as mp
import numpy as np
import socket

# 서버 IP 및 포트 설정
server_ip = '127.0.0.1'
server_port = 5054

# UDP 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind((server_ip, server_port))

print("서버가 시작되었습니다. 클라이언트의 메시지를 기다리는 중...")

try:
    while True:
        # 서버로부터의 응답 수신 (옵션)
        response, server_address = client_socket.recvfrom(1024)  # 1024는 버퍼 크기
        print(f"Received response from server: {response.decode()}")

        # 클라이언트에 응답 보내기
        response_message = "Hello, ai!"
        client_socket.sendto(response_message.encode(), server_address)
except KeyboardInterrupt:
    print('프로그램 종료')

finally:      
    # 소켓 닫기
    client_socket.close()