import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { FootervisiteurComponent } from './footer.component';

describe('FooterComponent', () => {
  let component: FootervisiteurComponent;
  let fixture: ComponentFixture<FootervisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ FootervisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(FootervisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
