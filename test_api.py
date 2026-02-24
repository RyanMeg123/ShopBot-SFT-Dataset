#!/usr/bin/env python3
"""
ShopBot API 测试和压测脚本
"""

import time
import asyncio
import httpx
import statistics
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000"


class APITester:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
    
    def test_health(self):
        """测试健康检查接口"""
        print("\n【测试1】健康检查 /health")
        response = self.client.get(f"{self.base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ 通过")
    
    def test_chat(self, message: str = "你好，这件T恤有什么颜色？"):
        """测试聊天接口"""
        print(f"\n【测试2】聊天接口 /chat")
        print(f"   Input: {message}")
        
        start = time.time()
        response = self.client.post(
            f"{self.base_url}/chat",
            json={"message": message, "temperature": 0.7}
        )
        elapsed = time.time() - start
        
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {data['response'][:60]}...")
        print(f"   Prompt tokens: {data['prompt_tokens']}")
        print(f"   Completion tokens: {data['completion_tokens']}")
        print(f"   耗时: {elapsed:.2f}s")
        assert response.status_code == 200
        print("   ✅ 通过")
        return elapsed
    
    def test_multiple_prompts(self):
        """测试多个prompt"""
        print("\n【测试3】多Prompt测试")
        prompts = [
            "你好，有什么优惠？",
            "这个鞋子太大了，想退",
            "我的订单什么时候到？",
            "现在有什么活动吗？",
        ]
        
        for prompt in prompts:
            print(f"\n   Prompt: {prompt}")
            response = self.client.post(
                f"{self.base_url}/chat",
                json={"message": prompt}
            )
            data = response.json()
            print(f"   → {data['response'][:50]}...")
        
        print("\n   ✅ 全部通过")
    
    def benchmark(self, num_requests: int = 10, concurrency: int = 1):
        """压测"""
        print(f"\n【压测】{num_requests}次请求，并发数{concurrency}")
        
        prompt = "你好，这件T恤有什么颜色？"
        latencies = []
        
        def make_request(_):
            start = time.time()
            try:
                response = self.client.post(
                    f"{self.base_url}/chat",
                    json={"message": prompt, "temperature": 0.7}
                )
                elapsed = time.time() - start
                return elapsed, response.status_code == 200
            except Exception as e:
                print(f"   Error: {e}")
                return time.time() - start, False
        
        start_total = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(make_request, range(num_requests)))
        
        total_time = time.time() - start_total
        latencies = [r[0] for r in results if r[1]]
        success_count = sum(1 for r in results if r[1])
        
        print(f"\n   总请求数: {num_requests}")
        print(f"   成功数: {success_count}")
        print(f"   失败数: {num_requests - success_count}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均延迟: {statistics.mean(latencies):.2f}s")
        print(f"   中位数延迟: {statistics.median(latencies):.2f}s")
        print(f"   最小延迟: {min(latencies):.2f}s")
        print(f"   最大延迟: {max(latencies):.2f}s")
        print(f"   吞吐量: {num_requests/total_time:.2f} req/s")
        
        if len(latencies) > 1:
            print(f"   标准差: {statistics.stdev(latencies):.2f}s")
    
    def close(self):
        self.client.close()


def main():
    print("=" * 60)
    print("🤖 ShopBot API 测试工具")
    print("=" * 60)
    print(f"API地址: {API_URL}")
    
    tester = APITester()
    
    try:
        # 基础测试
        tester.test_health()
        tester.test_chat()
        tester.test_multiple_prompts()
        
        # 压测（轻度）
        print("\n" + "=" * 60)
        print("开始压测...")
        print("=" * 60)
        
        tester.benchmark(num_requests=5, concurrency=1)  # 单并发
        tester.benchmark(num_requests=5, concurrency=2)  # 双并发
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.close()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
