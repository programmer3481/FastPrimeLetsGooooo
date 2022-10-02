package FastPrime;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryUtil.*;


public class Main {

    public static void main(String[] args) throws IOException {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long start = System.currentTimeMillis();
            int[] primes = calcPrimes(stack, 10000000);
            long finish  = System.currentTimeMillis();
            System.out.println(Arrays.toString(primes));
            System.out.println("took " + (finish - start) + "ms for whole function (not including print)");
        }
    }

    private static int[] calcPrimes(MemoryStack stack, int upTo) throws IOException {
        if (upTo < 1) {
            return new int[] {};
        }

        ArrayList<Integer> primes = new ArrayList<>(upTo);
        primes.add(2);

        if (upTo == 1) {
            return primes.stream().mapToInt(i->i).toArray();
        }

        IntBuffer ib = stack.mallocInt(1);
        PointerBuffer pb = stack.mallocPointer(1);

        IntBuffer isPrimeList = memCallocInt((upTo + 1) / 2 - 1);

        chErr(clGetPlatformIDs(null, ib));
        if (ib.get(0) == 0) {
            error("no openCL platforms found");
        }

        PointerBuffer platforms = stack.mallocPointer(ib.get(0));
        chErr(clGetPlatformIDs(platforms, (IntBuffer) null));

        chErr(clGetDeviceIDs(platforms.get(0), CL_DEVICE_TYPE_GPU, null, ib));
        if (ib.get(0) == 0) {
            error("no openCL devices found");
        }

        PointerBuffer devices = stack.mallocPointer(ib.get(0));
        chErr(clGetDeviceIDs(platforms.get(0), CL_DEVICE_TYPE_GPU, devices, (IntBuffer) null));

        long context = clCreateContext(null, devices, null, NULL, ib);
        chErr(ib.get(0));

        long commandQueue = clCreateCommandQueue(context, devices.get(0), 0, ib);
        chErr(ib.get(0));

        long isPListCLBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, isPrimeList.capacity()* 4L, ib);
        chErr(ib.get(0));

        long program = clCreateProgramWithSource(context, Files.readString(Path.of("./src/main/resources/prime.cl")), ib);
        chErr(ib.get(0));

        int buildErr = clBuildProgram(program, devices, stack.ASCII(""), null, 0);
        if (buildErr != CL_SUCCESS) {
            chErr(clGetProgramBuildInfo(program, devices.get(0), CL_PROGRAM_BUILD_LOG, (ByteBuffer) null, pb));
            ByteBuffer log = stack.malloc((int) pb.get(0));
            chErr(clGetProgramBuildInfo(program, devices.get(0), CL_PROGRAM_BUILD_LOG, log, null));
            System.out.println(memASCII(log));
            error(" ");
        }

        long kernel = clCreateKernel(program, "filterPrime", ib);
        chErr(ib.get(0));

        chErr(clSetKernelArg1p(kernel, 0, isPListCLBuf));
        chErr(clSetKernelArg1i(kernel, 1, upTo));

        int globalSize = (((int) Math.floor(Math.sqrt(upTo))) + 1) / 2 - 1;

        PointerBuffer globalWorkSize = stack.mallocPointer(1);
        globalWorkSize.put(0, globalSize);

        chErr(clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, globalWorkSize, null, null, null));

        chErr(clEnqueueReadBuffer(commandQueue, isPListCLBuf, true, 0, isPrimeList, null, null));

        chErr(clFlush(commandQueue));
        chErr(clFinish(commandQueue));

        for (int i = 0; i < isPrimeList.capacity(); i++) {
            if (isPrimeList.get(i) == 0) {
                primes.add((i+1)*2+1);
            }
        }

        chErr(clReleaseKernel(kernel));
        chErr(clReleaseProgram(program));
        chErr(clReleaseMemObject(isPListCLBuf));
        chErr(clReleaseCommandQueue(commandQueue));
        chErr(clReleaseContext(context));
        memFree(isPrimeList);

        return primes.stream().mapToInt(i->i).toArray();
    }

    private static void chErr(int code) {
        if (code != CL_SUCCESS) {
            error(String.format("OpenCL error [%d]", code));
        }
    }

    private static void error(String message) {
        throw new IllegalStateException(message);
    }
}
