; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vv.all()

define void @hardware_loop_probe(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 1024
  %4 = getelementptr float, ptr addrspace(6) %0, i64 2048
  br label %5

5:                                                ; preds = %aivscope.latch, %2
  %6 = phi i64 [ %45, %aivscope.latch ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %46

8:                                                ; preds = %5
  %9 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  br label %10

10:                                               ; preds = %42, %8
  %11 = phi i64 [ %43, %42 ], [ 0, %8 ]
  %12 = icmp slt i64 %11, %1
  br i1 %12, label %13, label %44

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %40, %13
  %15 = phi i64 [ %41, %40 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, %1
  br i1 %16, label %17, label %42

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %38, %17
  %19 = phi i64 [ %39, %38 ], [ 0, %17 ]
  %20 = icmp slt i64 %19, %1
  br i1 %20, label %21, label %40

21:                                               ; preds = %18
  br label %22

22:                                               ; preds = %25, %21
  %23 = phi i64 [ %37, %25 ], [ 0, %21 ]
  %24 = icmp slt i64 %23, %1
  br i1 %24, label %25, label %38

25:                                               ; preds = %22
  %26 = mul i64 %23, 64
  %27 = mul i64 %26, 4
  %28 = getelementptr i8, ptr addrspace(6) %0, i64 %27
  %29 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %28, i32 0, i32 0, i32 0)
  %30 = mul i64 %26, 4
  %31 = getelementptr i8, ptr addrspace(6) %3, i64 %30
  %32 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %31, i32 0, i32 0, i32 0)
  %33 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %29, <64 x float> %32, <256 x i1> %9)
  %34 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %33, <64 x float> %32, <256 x i1> %9)
  %35 = mul i64 %26, 4
  %36 = getelementptr i8, ptr addrspace(6) %4, i64 %35
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %34, ptr addrspace(6) %36, i32 0, i32 2, i32 0, <256 x i1> %9)
  call void @llvm.hivm.mem.bar.vv.all()
  %37 = add i64 %23, 1
  br label %22

38:                                               ; preds = %22
  call void @llvm.hivm.mem.bar.vv.all()
  %39 = add i64 %19, 1
  br label %18

40:                                               ; preds = %18
  call void @llvm.hivm.mem.bar.vv.all()
  %41 = add i64 %15, 1
  br label %14

42:                                               ; preds = %14
  call void @llvm.hivm.mem.bar.vv.all()
  %43 = add i64 %11, 1
  br label %10

44:                                               ; preds = %10
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %44
  %45 = add i64 %6, 1
  br label %5, !llvm.loop !1

46:                                               ; preds = %5
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
