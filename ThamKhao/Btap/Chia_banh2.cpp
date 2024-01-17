//#include <iostream>
//#include<algorithm>
//using namespace std;
//
//long long Chia(long long a, long long b, long long n){
//	int count = 0;
//	
//	long long MAX, MIN;
//	MAX = max(a,b);
//	MIN = min(a,b);
//	
//	for(int i = MAX; i < n; i += MAX){
//		if((n-i) % MIN == 0){
//			count++;
//		}
//	} return count;

//	Cach2
//	int count = 0;
//	
//	long long MAX, MIN;
//	MAX = max(a,b);
//	MIN = min(a,b);
//	
//	for(int i = n-1; i >= MAX; i--){
//		if(i % a == 0 && i % b == 0) {
//			count++;
//		}
//	} 
//	return count;
//}
//
//int main()
//{
//	
//	long long a, b, n;
//	cin >> a;
//	cin >> b;
//	cin >> n;
//	
//	
//	cout << Chia(a,b,n);
//return 0;
//}


#include<bits/stdc++.h>

using namespace std;


int main(){
	
	int a,b,n;
	cin>>a>>b>>n;
	if(a==1 && b==1 && n%2==0) cout<< n-1;
	if(a%2==0 && b!=0 && n%2!=0) cout<<0;
	int Max =max(a,b);
	int Min = min(a,b);
	int count =0;
	for(int i = Max ;i<n;i+=Max){
		if((n-i)%Min==0) count++;
	}
	cout<<count;
	
}

