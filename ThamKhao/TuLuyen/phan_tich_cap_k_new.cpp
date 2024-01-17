#include<bits/stdc++.h>
using namespace std;

int phan_tich_k(int n,int i,int tmp,int k){
//	cout<<"f("<<n<<","<<i<<","<<tmp<<") ";
	if(n<i){
		if(n==0&&tmp>=k){
//			cout<<endl;
			return 1;
		} 
		
		return 0;
	} 
	return phan_tich_k(n-i,i,tmp*i,k)+phan_tich_k(n,i+1,tmp,k);
}

int main(){
	int n,k;
	cout<<"Nhap N = ";cin>>n;
	cout<<"Nhap K = ";cin>>k;
	clock_t time=clock();
	cout<<phan_tich_k(n,1,1,k)<<endl;
	cout<<clock()-time<<"ms";
}
