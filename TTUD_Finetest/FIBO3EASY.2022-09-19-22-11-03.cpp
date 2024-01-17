#include <iostream>
#include <map>
using namespace std;
map <int, long long> f;
/*
f0 = 0
f1 =1
f2 = 2
f(3k) = f(2k) k > 0
f(3k+1) = f(2k) + f(2k+1)
f(3k+2) = f(2k) + f(2k+1) +f (2k+2)

k =1 , f3 = f2 = 2
		f4 = f2 + f3 = 4
		f5 = f2 + f3 + f4 = 8
		f6 = f4 = f2 + f3 = 4
		f7 = f4 + f5 = f2 + f3 + f2 + f3 + f4 = 12
		f8 = f4 + f5 + f6 = (f2 + f3) + (f2 + f3 + f4) + (f2 + f3) = 16 
		f9 = f6 = f4 = f2 + f3 = 4
		f10 = f6 + f7 = 16
		
		
*/
long long fn(long long n){
	if(n<=2) return f[n];
	else{
		if(f[n] == 0){
			int r = n%3;
			int k = n/3;
			for(int i=0;i<=r;i++){
				f[n] += fn(2*k +i);
			}
		}
		
	}
	return f[n];
}
//long long fn ( long n){
//	long long temp;
//	if(n==0) return 0;
//	if(n==1) return 1;
//	if(n==2) return 2;
//	if(n%3==0)
//	{
//		
//		temp = n/3*2;
//		return fn(temp);
//	 } 
//	if(n%3==1){
//		temp = (n-1)/3*2;
//		return fn(temp) + fn(temp + 1);
//	}
//	if(n%3==2){
//		temp = (n-2)/3*2;
//		return fn(temp) + fn(temp + 1) + fn(temp + 2);
//	}
//}
//1923245  18769118016
int main()
{
	long long n ;
	f[0] = 0;
	f[1] = 1;
	f[2] = 2;
	cout<<"Nhap n = "; cin>>n;
	while(n<=0){
		cout<<"Nhap n = "; cin>>n;
	}
	cout<<"f("<<n<<") = "<<fn(n);
return 0;
}

