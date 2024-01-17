#include <iostream>
#include<string>
using namespace std;

//const int N = 3;
//const int W = 19;
//
//int a[] = {14, 10, 6};
//int b[] = {20,16,8};
//
//int f[N+1][W+1], k, n;

const int Max = 1000;
string s = "abbca";
long long f[Max] = {1}, t=0;

int main()
{
//	for(h=0; h <= W; h++)	f[0][h] = 0;
//	for(k=1; k<=N; k++)
//		for(h=0; h <=W; h++){
//			f[k][h] = f[k-1][h];
//			if(h >=)
//		}

	for(int p = 1; p<s.length(); p++){
		f[p] = 1;
		for(int k = p-1; k>=0; k--){
			f[p] += f[k];
			if(s[p] ==s[k]){
				f[p] = f[p]-1;
				break;
			}
		}
		t+=f[p];
	}
	cout << "So chuoi con phan biet: " << t;
return 0;
}

