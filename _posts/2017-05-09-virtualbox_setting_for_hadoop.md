---
title: Hadoop ecosystem 경험하기-(1)
author: "Seongbong Kim"
date: 2017-05-09 11:17:00 +0900
categories: jekyll update study hadoop
permalink: /blog/:title
comments: true
---

아직은 Windows가 익숙하시지만 Hadoop ecosystem을 경험/학습하고 싶으신 분들을 위해 제가 VirtualBox로 Hadoop ecosystem (~~삽질~~)구성 했던 내용을 갈고 닦아 설정 과정을 기록으로 남깁니다. 아쉽게도 학습용이기 때문에 방화벽 설정 등 보안 관련된 사항등은 안전하지 않은 방법인 서비스 Off/포트 전체 Open 등으로 진행되었습니다.

이번 포스트는 Hadoop 설치 전 네트워크 환경 구성에 관련된 내용부터 시작합니다. Hadoop은 네트워크를 기반으로 통신하며 병렬처리하는 시스템이기 때문에 네트워크 구성이 제대로 안되면 아무것도 안됩니다.

#### 컴퓨터 환경
- Intel i7 6700 (Skylake)
- Memory 32GB DDR4
- HDD 1TB
- Windows 10 64 bit
- PC 인터넷 : 공유기를 이용한 사설 IP 할당 가능 환경
<br>

#### 사전 준비사항
- [VirtualBox CentOS 설치 (매뉴얼-제타위키)](http://vault.centos.org/6.7/isos/x86_64/)
    ※ [VirtualBox Platform package + Extension Pack 설치 ](https://www.virtualbox.org/wiki/Downloads)
    ※ [CentOS 6.7 minimal iso (64bit)](http://vault.centos.org/6.7/isos/x86_64/)
<br>

#### 구성 방안
현재까지 제가 구성한 VirtualBox 가상머신 구성도입니다.
<br>

![구성도](/assets/virtualbox/my_ip_config.png)

공유기를 통해 가상머신 별로 IP를 Host와 동일한 레벨로 할당하였기 때문에 가상머신을 동작시킨 이후에는 마치 한 허브에 물려있는 네트워크처럼 가상머신을 사용할 수 있다는 장점이 있습니다. (~~마치 여러대의 컴퓨터를 허브에 물려놓은 느낌~~)

Hadoop 구성에서는 CenOS1~4로 명명된 가상머신 4대를 사용하고 향후에 별도 포스트로 R/Python/Shiny/TensorFlow 등을 사용할 가상머신도 구성을 해보겠습니다.(언제쯤 가능할 지...)
<br>

#### [VirtualBox] 가상머신 추가 생성

이 포스트에서는 CentOS설치된 가상머신에 대해서 Hadoop 구성을 할 예정이므로, CentOS 가상머신 4대가 필요합니다.

우선 CentOS 설치를 1대에서 진행하고, 기본적인 설치를 마친 후 가상머신을 3번 복제하는 방식으로 4대를 준비합니다.
- 개별 가상머신 환경
    - CentOS 6.7
    - CPU 2개
    - Memory 4GB
    - HDD 100GB
<br>

#### [VirtualBox] 네트워크 어댑터 설정
가상 머신 생성 시 네트워크 어댑터는 기본 NAT로 설정되어 있는데, 이를 브리지 어댑터로 변경하면 위에 제가 구성한 것과 동일한 방식으로 IP를 할당하여 네트워크를 구성할 수 있습니다.

- VirtualBox 실행
- 가상머신 우클릭 -- 설정
- 네트워크 -- 어댑터1 -- 다음에 연결됨(A): **브리지 어댑터**
  (혹시 이 네트워크 어댑터를 먼저 설정하신 후 가상머신을 복제하는 경우에는 네트워크 어댑터의 MAC주소도 복사되므로 위 메뉴에서 MAC주소 옆의 새로고침 버튼을 눌러 새로운 MC주소를 받으세요)
<br>

#### 고정 IP 설정
이 부분부터는 각 가상머신을 실행시킨 후에 각각 설정을 해줘야 하는 부분입니다.

우선 가상머신을 실행시키고 root로 로그인을 합니다.

CentOS에서 network 설정을 관리하는 파일은 `/etc/sysconfig/network-scripts/` 폴더 안에 `ifcfg-eth?`형식으로 들어있습니다.

저는 아래와 같이 조회하면 `ifcfg-eth0` 파일이 있어서 이 파일을 vi 편집기로 수정하도록 하겠습니다.
```
# vi /etc/sysconfig/network-scripts/ifcfg-eth0
```
```
ONBOOT=yes
NM_CONTROLLED=yes
IPADDR=192.168.0.101
NETMASK=255.255.255.0
GATEWAY=192.168.0.1
DNS1=8.8.8.8
DNS2=8.8.4.4
```

>CentOS를 설치하자마자 이 부분을 따라하신다면 ifcf-eth0이시겠지만 eth1이나 eth2와 같이 다른 이름으로 파일이 존재할 수도 있습니다. vi 편집기로 파일을 열었을 때 `HWADDR=~~~`와 같은 줄이 있다면, 가상머신 실행전 네트워크 설정에서 MAC 주소를 메모한 후 들어오셔서 `HWADDR=MAC주소`가 일치하는 파일을 수정하시면 됩니다.

`i`를 누른 후 내용을 수정하고 `Esc`를 누른 후 `:wq`를 입력하고 `Enter`를 누르면 수정 내용을 저장하고 빠져나옵니다. `:q!`를 입력하면 수정사항 변경없이 vi편집기를 종료합니다.

아래는 제가 가진 가상머신 이름과 IP, 추후에 설정할 HOSTNAME이 정리된 표 입니다.
|VM Name|IP|HOSTNAME
|:-:|:-:|:-:|
|CentOS1| 192.168.0.101 | oops1|
|CentOS2| 192.168.0.102 | oops2|
|CentOS3| 192.168.0.103 | oops3|
|CentOS4| 192.168.0.104 | oops4|

<br>

#### hosts 편집 (네트워크 이름 설정)
네트워크끼리 통신을 하려면 상대방 네트워크의 IP를 입력하는 일이 잦은데 이를 일일이 작성하기에는 번거롭기에 네트워크의 이름을 설정해주면 이 이름으로 네트워크 명령어를 내릴 수 있게 됩니다. 이를 관리하는 파일은 `etc` 폴더 안의 `hosts` 파일입니다.

내용을 수정해야 하니 다음과 같이 vi 편집기를 실행시킵니다.
`# vi /etc/hosts`

아마 파일 내에 `127.0.0.1  localhost localhost.~~~` 등의 내용이 있을텐데 내용을 수정하지 말고 실행시킨 파일의 가장 밑 줄에 아래 내용을 추가합니다.
```
192.168.0.101    oops1
192.168.0.102    oops2
192.168.0.103    oops3
192.168.0.104    oops4
```

#### SSH 키 생성

hadoop은 datanode로 지정된 PC와 ssh를 기반으로 통신을 주고 받는데 ssh로 접속할 때마다 password를 묻게 되면 골치 아파집니다. 그래서 ssh로 접속할 때 password를 묻지 않는 신뢰할 수 있는 PC로 서로 등록하는 과정이 필요합니다.

우선 oops1(NameNode)에서 인증키를 생성하고 이를 `authorized_keys`로 복사해 줍니다.
```bash
oops1$ ssh-keygen -t rsa
oops1$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
oops1$ chmod 0600 ~/.ssh/authorized_keys
```

ssh로 oops2, oops3, oops4에 접속하고 인증키를 authorized_keys에 등록합니다.
```
oops1$ ssh hadoop@oops2 cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
oops1$ ssh hadoop@oops3 cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
oops1$ ssh hadoop@oops4 cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

이렇게 만들어진 authorized_keys를 oops2~4에 복사를 해주면 oops1에서 oops2~4에 password 없이 접속이 가능해집니다.
```
oops1$ scp -rp authorized_keys hadoop@oops2:~/.ssh/authorized_keys
oops1$ scp -rp authorized_keys hadoop@oops3:~/.ssh/authorized_keys
oops1$ scp -rp authorized_keys hadoop@oops4:~/.ssh/authorized_keys
```

Java / Hadoop / Hive 등 설치하는 과정은 다음 글에 포스팅 하겠습니다.
