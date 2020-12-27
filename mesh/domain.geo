Point(1) = {0,0,0,1};
//+
Point(2) = {2,0,0,1};
//+
Point(3) = {2,1,0,1};
//+
Point(4) = {2,3,0,1};
//+
Point(5) = {1,3,0,1};
//+
Point(6) = {0,3,0,1};
//+
Line(1) = {1,2};
//+
Line(2) = {2,3};
//+
Line(3) = {3,4};
//+
Line(4) = {4,5};
//+
Line(5) = {5,6};
//+ 
Line(6) = {6,1};
//+
Line Loop(7) = {1,2,3,4,5,6};
//+
Plane Surface(8) = {7}; 
//+
Physical Curve("DirichletBdry") = {2,-4};
//+
Physical Curve("NeumannBdry") = {1,3,-5,-6};
//+
Physical Surface("Domain") = {8};