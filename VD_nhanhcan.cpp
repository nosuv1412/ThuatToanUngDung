#include <iostream>
using namespace std;

const int v =10000;

int n = 8, z=10000000, a[100], cmin=1;
int b[]= {0,0,0,0,0,0,0,0}; //danh dau xem da tham dinh i hay chua

int c[][8] = {
	{0,5,1,2,v,v,v,v},
	{5,0,2,v,2,v,v,v},
	{1,2,0,2,4,3,6,v},
	{2,v,2,0,v,v,2,v},
	{v,2,4,v,0,2,v,3},
	{v,v,3,v,2,0,3,2},
	{v,v,6,2,v,3,0,2},
	{v,v,v,v,3,2,2,0}
};

long long dem = 0;

void in_ket_qua(int t){
	dem++;
	if(t > z)return;
	z=t;
	cout << "Duong di thu " << dem << ". Tim duoc duong di moi, tong chi phi = " << z << endl;
	for(int i = 0; i< n; i++) cout << a[i]+1 << "->";
	cout<< 1 << endl;
	
}

// tinh xem tham dinh a[k] nao, t = tong khoi luong da thuc hien
void tham(int k, int t) {
	if(t>= z) return;
	if(k==n){
		t += c[a[k-1]][0];
		in_ket_qua(t);
		return;
	}
	for(int i =0; i < n; i++) {
		if(b[i] == 0) {
			b[i] = 1;
			a[k] = i; tham(k+1, t +c[a[k-1]][i]);
			b[i] = 0;
		}
	}
}


int main()
{
	a[0]= 0, b[0]=1;
	tham(1,0);
	cout << "So duong di xem xet: " << dem << ", ghi nhan duong di ngan nhat = " << z;
return 0;
}

