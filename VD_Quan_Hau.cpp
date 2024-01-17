#include <iostream>
using namespace std;

int n, a[100], dem;

void in_ban_co(){
	cout <<"Phuong an " << dem++ << ":" << endl;

}
void dat_ok(int k, int cot){
	for(int i=1; i<k;i++)
		if((cot == a[i]) || (k-cot == i-a[i]) || (k+cot == i+a[i]))
			return false;
		return true;
}

void dat(int k){
	if(k>n)
		in_ban_co(); return;
		
	for(int i =1; i<=n; i++){
		if(dat_ok(k,i)) {
			a[k]=i; dat(k+1);
		}
	}
}

int main()
{
	cout << "N="; cin >>n;
	dat(1);
	cout << "Co tong cong " << dem << "phuong an";

return 0;
}

