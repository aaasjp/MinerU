"""
测试 MinerU FastAPI /file_parse 接口
"""
import os
import requests
import json
from pathlib import Path

# API 服务器地址
API_BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = f"{API_BASE_URL}/file_parse"

# 测试文件路径
DEMO_PDFS_DIR = Path(__file__).parent / "pdfs"

backend = "vlm-mlx-engine"
output_dir = "/Users/ailabuser7-1/Documents/test_output"


def test_file_parse_json_response():
    """测试 /file_parse 接口 - JSON 响应格式"""
    print("\n=== 测试 1: JSON 响应格式 ===")
    
    pdf_path = DEMO_PDFS_DIR / "demo2.pdf"
    if not pdf_path.exists():
        print(f"错误: 测试文件不存在: {pdf_path}")
        return
    
    with open(pdf_path, "rb") as f:
        files = [("files", (pdf_path.name, f.read(), "application/pdf"))]
        
        data = {
            "output_dir": output_dir,
            "lang_list": ["ch"],
            "backend": backend,
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": True,
            "return_model_output": True,
            "return_content_list": True,
            "return_images": True,
            "response_format_zip": False,
            "f_draw_layout_bbox": True,
            "f_draw_span_bbox": True,
            "f_dump_orig_pdf": True,
            "start_page_id": 0,
            "end_page_id": 99999,
        }
        
        try:
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=300)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"后端: {result.get('backend')}")
                print(f"版本: {result.get('version')}")
                print(f"结果数量: {len(result.get('results', {}))}")
                
                for pdf_name, pdf_result in result.get("results", {}).items():
                    print(f"\n文件: {pdf_name}")
                    if "md_content" in pdf_result:
                        md_len = len(pdf_result["md_content"]) if pdf_result["md_content"] else 0
                        print(f"  Markdown 内容长度: {md_len} 字符")
                        if md_len > 0:
                            print(f"  Markdown 预览 (前200字符): {pdf_result['md_content'][:200]}...")
            else:
                print(f"错误响应: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到 API 服务器，请确保服务器已启动")
        except Exception as e:
            print(f"错误: {e}")


def test_file_parse_zip_response():
    """测试 /file_parse 接口 - ZIP 响应格式"""
    print("\n=== 测试 2: ZIP 响应格式 ===")
    
    pdf_path = DEMO_PDFS_DIR / "【高级ai算法工程师_青岛 】唐小宇 9年.pdf"
    if not pdf_path.exists():
        print(f"错误: 测试文件不存在: {pdf_path}")
        return
    
    with open(pdf_path, "rb") as f:
        files = [("files", (pdf_path.name, f.read(), "application/pdf"))]
        
        data = {
            "output_dir": output_dir,
            "lang_list": ["ch"],
            "backend": backend,
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": True,
            "return_model_output": True,
            "return_content_list": True,
            "return_images": True,
            "response_format_zip": True,
            "f_draw_layout_bbox": True,
            "f_draw_span_bbox": True,
            "f_dump_orig_pdf": True,
            "start_page_id": 0,
            "end_page_id": 99999,
        }
        
        try:
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=300)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                print(f"Content-Type: {content_type}")
                
                if "application/zip" in content_type:
                    zip_path = Path(output_dir) / "test_results.zip"
                    with open(zip_path, "wb") as zip_file:
                        zip_file.write(response.content)
                    print(f"ZIP 文件已保存到: {zip_path}")
                    print(f"ZIP 文件大小: {len(response.content)} 字节")
                else:
                    print(f"意外的内容类型: {content_type}")
            else:
                print(f"错误响应: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到 API 服务器，请确保服务器已启动")
        except Exception as e:
            print(f"错误: {e}")


def test_file_parse_multiple_files():
    """测试 /file_parse 接口 - 多文件上传"""
    print("\n=== 测试 3: 多文件上传 ===")
    
    pdf_files = ["small_ocr.pdf", "demo1.pdf"]
    files = []
    
    for pdf_file in pdf_files:
        pdf_path = DEMO_PDFS_DIR / pdf_file
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                files.append(("files", (pdf_path.name, f.read(), "application/pdf")))
        else:
            print(f"警告: 文件不存在: {pdf_path}")
    
    if not files:
        print("错误: 没有可用的测试文件")
        return
    
    data = {
        "output_dir": output_dir,
        "lang_list": ["ch", "ch"],  # 为每个文件指定语言
        "backend": backend,
        "parse_method": "auto",
        "formula_enable": True,
        "table_enable": True,
        "return_md": True,
        "return_middle_json": False,
        "return_model_output": False,
        "return_content_list": False,
        "return_images": False,
        "response_format_zip": False,
        "f_draw_layout_bbox": False,
        "f_draw_span_bbox": False,
        "f_dump_orig_pdf": False,
        "start_page_id": 0,
        "end_page_id": 99999,
    }
    
    try:
        response = requests.post(API_ENDPOINT, files=files, data=data, timeout=600)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"处理了 {len(result.get('results', {}))} 个文件")
            
            for pdf_name, pdf_result in result.get("results", {}).items():
                print(f"\n文件: {pdf_name}")
                if "md_content" in pdf_result:
                    md_len = len(pdf_result["md_content"]) if pdf_result["md_content"] else 0
                    print(f"  Markdown 内容长度: {md_len} 字符")
        else:
            print(f"错误响应: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到 API 服务器，请确保服务器已启动")
    except Exception as e:
        print(f"错误: {e}")


def test_file_parse_with_images():
    """测试 /file_parse 接口 - 返回图片"""
    print("\n=== 测试 4: 返回图片 ===")
    
    pdf_path = DEMO_PDFS_DIR / "small_ocr.pdf"
    if not pdf_path.exists():
        print(f"错误: 测试文件不存在: {pdf_path}")
        return
    
    with open(pdf_path, "rb") as f:
        files = [("files", (pdf_path.name, f.read(), "application/pdf"))]
        
        data = {
            "output_dir": output_dir,
            "lang_list": ["ch"],
            "backend": backend,
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": False,
            "return_model_output": False,
            "return_content_list": False,
            "return_images": True,  # 启用图片返回
            "response_format_zip": False,
            "start_page_id": 0,
            "end_page_id": 99999,
        }
        
        try:
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=300)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                for pdf_name, pdf_result in result.get("results", {}).items():
                    print(f"\n文件: {pdf_name}")
                    if "images" in pdf_result:
                        image_count = len(pdf_result["images"])
                        print(f"  提取的图片数量: {image_count}")
                        for img_name in list(pdf_result["images"].keys())[:3]:  # 只显示前3个
                            img_data = pdf_result["images"][img_name]
                            print(f"    图片: {img_name} (base64长度: {len(img_data)} 字符)")
            else:
                print(f"错误响应: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到 API 服务器，请确保服务器已启动")
        except Exception as e:
            print(f"错误: {e}")


def test_file_parse_vlm_backend():
    """测试 /file_parse 接口 - VLM 后端"""
    print("\n=== 测试 5: VLM 后端 ===")
    
    pdf_path = DEMO_PDFS_DIR / "small_ocr.pdf"
    if not pdf_path.exists():
        print(f"错误: 测试文件不存在: {pdf_path}")
        return
    
    with open(pdf_path, "rb") as f:
        files = [("files", (pdf_path.name, f.read(), "application/pdf"))]
        
        data = {
            "output_dir": output_dir,
            "lang_list": ["ch"],
            "backend": backend,
            "formula_enable": True,
            "table_enable": True,
            "return_md": True,
            "return_middle_json": False,
            "return_model_output": False,
            "return_content_list": False,
            "return_images": False,
            "response_format_zip": False,
            "start_page_id": 0,
            "end_page_id": 99999,
        }
        
        try:
            response = requests.post(API_ENDPOINT, files=files, data=data, timeout=600)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"后端: {result.get('backend')}")
                print(f"版本: {result.get('version')}")
                
                for pdf_name, pdf_result in result.get("results", {}).items():
                    print(f"\n文件: {pdf_name}")
                    if "md_content" in pdf_result:
                        md_len = len(pdf_result["md_content"]) if pdf_result["md_content"] else 0
                        print(f"  Markdown 内容长度: {md_len} 字符")
            else:
                print(f"错误响应: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("错误: 无法连接到 API 服务器，请确保服务器已启动")
        except Exception as e:
            print(f"错误: {e}")


def check_server_status():
    """检查服务器状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✓ API 服务器运行正常")
            return True
    except:
        pass
    
    print("✗ API 服务器未运行或无法访问")
    print(f"  请先启动服务器: mineru-api --host 127.0.0.1 --port 8000")
    return False


def main():
    """主函数"""
    print("=" * 60)
    print("MinerU FastAPI /file_parse 接口测试")
    print("=" * 60)
    
    if not check_server_status():
        return
    
    # 运行所有测试
    #test_file_parse_json_response()
    test_file_parse_zip_response()
    #test_file_parse_multiple_files()
    #test_file_parse_with_images()
    #test_file_parse_vlm_backend()  # VLM 后端测试可能需要较长时间，注释掉
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

