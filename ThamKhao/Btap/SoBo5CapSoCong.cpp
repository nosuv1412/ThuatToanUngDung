#include <iostream>
#include <algorithm>
using namespace std;

const int MAX = 100;
int a[MAX], n;

bool check(int k) {
	for (int i = 0; i < n; i++)
		if (a[i] == k)
			return true;
	return false;
}
// cap so cong VD: co u1=3 cong sai d=3 --> 3,6,9,12,15 la 1 cap so cong
int socach() {
	int count = 0;
	for (int i = 0; i < n ; i++)
		for (int j = i + 1; j < n ; j++) {
			
			int socuoi = a[j];
			int sodau  = a[i];
			int d = 2; // lay 2 so dau va cuoi 
			int khoangcach = socuoi - sodau; // cong sai cua 2 so dau (Khoang cach)
			
			for (int k = 0; k < 3; k++) {
				//check xem so tiep theo(SoCuoi + CongSai) co trong mang ko --> co thi thay doi SoCuoi va d 
				if (check(socuoi + khoangcach)) {
					socuoi = socuoi + khoangcach;
					d++;
				} else break;
				if (d == 5) count++; // neu du 5 phan tu thi count + 1;
			}
		}
	return count;
}
int main() {
	cin >> n;
	for (int i = 0; i < n; i++)
		cin >> a[i];
	//Sap xem lai mang a;
	sort(a , a + n);
	cout <<socach();
	return 0;
}
