#include<iostream>
#include<vector>
using namespace std;
const int MAX=100;
int a[MAX],b[MAX], n,m,count = 0;
vector<int> c;
int main(){
	cout <<"Nhap m = ";
	cin >>m;
	cout <<"Nhap n = ";
	cin >>n;
	n = n-1;
	for(int i=0;i<=n;i++){
		int x;
		cout <<"a["<<i<<"] = ";
		cin >>x;
		if(x<m){
			int length = c.size();
			for(int j=0;j<length;j++){
				if(c[j]<m){
					int s = c[j]+x;
					if(s==m)
						count++;
					if(s<m)
						c.push_back(s);
				}
			}
			c.push_back(x);
		}
		if(x == m)
			count++;
	}
	cout <<"Co tat ca "<<count<<" cach phan tich.";
}
