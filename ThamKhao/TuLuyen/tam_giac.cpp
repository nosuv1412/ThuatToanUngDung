#include<bits/stdc++.h>
using namespace std;


int main(){
	int n;
	long long kq=0;
	cin>>n;
//	clock_t time =clock();
	for(int i=1;i<sqrt(n);i++){
		for(int j=i+1;j<sqrt(n);j++){
			if(j*j+2*j*i>n) break;
			if(__gcd(i,j)==1&&(i+j)%2==1){
				int a=j*j-i*i;
				int b=2*i*j;
				int c=j*j+i*i;
				int tmp1=a,tmp2=b,tmp3=c;
				int index=1;
				while(tmp1+tmp2+tmp3<=n){
					kq++;
					index++;
					tmp1=a*index;
					tmp2=b*index;
					tmp3=c*index;
				}
			}
		}
	}
//	cout<<clock()-time<<"ms"<<endl;
	cout<<kq;
}

//Cho so nguyen N(1<=N<100000). Dem xem co bn tam giac vuong
//co cac canh do dai nguyen va chi vi k vuot qua N


