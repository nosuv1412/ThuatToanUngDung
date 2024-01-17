#include<bits/stdc++.h>
using namespace std;

long long f(int n,int i){
	if(n==0) return 1;
	if(n<i) return 0;
	return f(n,i+1)+f(n-i,i); 
}
int main(){
	int n;cout<<"nhap n"<<endl;
	cin>>n;
//	clock_t time=clock();
	cout<<"so kieu phan tich la: "<<f(n,1)-1;
//	cout<<clock()-time<<"ms";
}

//#include<bits/stdc++.h>
//
//using namespace std;
//long long a[1000];
//long long S(int n){
//	a[0] = 1;
//	for(int i =1;i<n;i++){
//		for(int j=1;j<=n;j++){
//			if(j>=i)
//				a[j] +=a[j-i];
//			cout<<a[j]<<" ";
//		}
//		cout<<endl;
//	}
//	return a[n];
//}
//int main(){
//	int n;
//	cin>>n;
//	for(int i =0;i<1000;i++) a[i] =0; 
//	cout<<S(n);
//}
